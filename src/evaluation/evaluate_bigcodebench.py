"""
Evaluation for BigCodeBench outputs.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data_processing.schemas import InferenceOutputExample
from tqdm import tqdm

# BigCodeBench eval returns lowercase; we emit MemRL-aligned uppercase status.
_BCB_PASS = "pass"
_BCB_FAIL = "fail"
_BCB_TIMEOUT = "timeout"

_MME_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BCB_REPO = _MME_ROOT / "data" / "bigcodebench" / "bigcodebench-main"
_INFRA_ERROR_PATTERNS = (
    "No module named '__test__'",
    "qt.qpa.xcb",
    "Could not load the Qt platform plugin",
)

# Code patterns that can kill the parent evaluator (e.g. pkill, killall).
_DANGEROUS_CODE_PATTERNS = (
    "pkill",
    "killall",
    "os.kill(",
    "os.killpg(",
    "signal.SIGKILL",
)


def _ensure_bigcodebench_on_path(bcb_repo: str | None = None) -> str:
    repo = Path(bcb_repo).expanduser().resolve() if bcb_repo else _DEFAULT_BCB_REPO.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"BigCodeBench repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return str(repo)


def _sanitize_code(code: str, entry_point: str, bcb_repo: str | None = None) -> str:
    _ensure_bigcodebench_on_path(bcb_repo)
    try:
        from bigcodebench.sanitize import sanitize

        return sanitize(code, entry_point)
    except Exception:
        return code


_SUBPROCESS_EVAL_SCRIPT = r'''
import json, os, sys, tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

payload = json.loads(sys.stdin.read())
bcb_repo = payload["bcb_repo"]
if bcb_repo and bcb_repo not in sys.path:
    sys.path.insert(0, bcb_repo)

from bigcodebench.eval import untrusted_check

try:
    stat, details = untrusted_check(
        code=payload["code"],
        test_code=payload["test_code"],
        entry_point=payload["entry_point"],
        max_as_limit=payload["max_as_limit"],
        max_data_limit=payload["max_data_limit"],
        max_stack_limit=payload["max_stack_limit"],
        min_time_limit=payload["min_time_limit"],
        gt_time_limit=payload["gt_time_limit"],
    )
    print(json.dumps({"status": "ok", "stat": stat, "details": dict(details) if details else {}}))
except Exception as e:
    print(json.dumps({"status": "err", "error": str(e)}))
'''


def _run_untrusted_check_with_hard_timeout(
    *,
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: int = 30 * 1024,
    max_data_limit: int = 30 * 1024,
    max_stack_limit: int = 10,
    min_time_limit: float = 1.0,
    gt_time_limit: float = 60.0,
    hard_timeout_s: float = 120.0,
    bcb_repo: str | None = None,
) -> Tuple[Any | None, Any | None, str | None, bool]:
    import subprocess as _sp

    bcb_path = str(_DEFAULT_BCB_REPO.resolve()) if bcb_repo is None else bcb_repo
    payload = json.dumps({
        "code": code,
        "test_code": test_code,
        "entry_point": entry_point,
        "max_as_limit": max_as_limit,
        "max_data_limit": max_data_limit,
        "max_stack_limit": max_stack_limit,
        "min_time_limit": min_time_limit,
        "gt_time_limit": gt_time_limit,
        "bcb_repo": bcb_path,
    })

    # Block SIGTERM while child runs — rogue code (e.g. pkill -f python)
    # sends SIGTERM to all matching processes including the parent.
    # signal.signal() only works in the main thread; skip when running in a
    # worker thread (the child is already isolated via start_new_session=True).
    import threading
    _is_main_thread = threading.current_thread() is threading.main_thread()
    _prev_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN) if _is_main_thread else None
    try:
        proc = _sp.Popen(
            [sys.executable, "-c", _SUBPROCESS_EVAL_SCRIPT],
            stdin=_sp.PIPE,
            stdout=_sp.PIPE,
            stderr=_sp.PIPE,
            start_new_session=True,  # fully isolate process group
        )
        stdout, _ = proc.communicate(input=payload.encode(), timeout=hard_timeout_s)
    except _sp.TimeoutExpired:
        # Kill the entire process group of the child.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=5)
        return None, None, f"Hard timeout after {hard_timeout_s}s", True
    except Exception as e:
        return None, None, str(e), False
    finally:
        if _is_main_thread:
            signal.signal(signal.SIGTERM, _prev_sigterm)

    if proc.returncode != 0:
        return None, None, f"subprocess exited with code {proc.returncode}", False

    try:
        result = json.loads(stdout.decode().strip().splitlines()[-1])
    except Exception as e:
        return None, None, f"failed to parse subprocess output: {e}", False

    if result.get("status") == "ok":
        return result["stat"], result.get("details"), None, False
    return None, None, result.get("error", "unknown"), False


def _load_outputs(path: str) -> List[InferenceOutputExample]:
    outputs: List[InferenceOutputExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                outputs.append(InferenceOutputExample.from_dict(json.loads(line)))
    return outputs


def _evaluate_one(out: InferenceOutputExample, bcb_repo: str | None = None) -> Dict[str, Any]:
    meta = out.metadata
    task_id = str(meta.get("task_id") or out.id)
    entry_point = str(meta.get("entry_point") or "task_func")
    test_code = str(meta.get("test") or "")
    code = str(meta.get("code") or "")

    if not code.strip():
        return {"task_id": task_id, "status": "RUNTIME_ERROR", "error": "empty_code", "label": False}
    if not test_code.strip():
        return {"task_id": task_id, "status": "SYNTAX_OK", "error": "no_test_code", "label": False}

    # Skip code that can kill the evaluator process.
    for pat in _DANGEROUS_CODE_PATTERNS:
        if pat in code:
            return {"task_id": task_id, "status": "FAIL", "error": f"dangerous_code_pattern: {pat}", "label": False}

    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        return {"task_id": task_id, "status": "SYNTAX_ERROR", "error": str(e), "label": False}

    clean_code = _sanitize_code(code, entry_point, bcb_repo=bcb_repo)
    stat, details, err, hard_timed_out = _run_untrusted_check_with_hard_timeout(
        code=clean_code,
        test_code=test_code,
        entry_point=entry_point,
        gt_time_limit=60.0,
        hard_timeout_s=120.0,
        bcb_repo=bcb_repo,
    )
    if hard_timed_out:
        return {"task_id": task_id, "status": "TIMEOUT", "error": err or "hard_timeout", "label": False}
    if err:
        err_str = str(err)
        if any(pat in err_str for pat in _INFRA_ERROR_PATTERNS):
            raise RuntimeError(
                "BigCodeBench evaluation infrastructure error "
                f"(task_id={task_id}): {err_str}"
            )
        return {"task_id": task_id, "status": "RUNTIME_ERROR", "error": err, "label": False}
    if stat == _BCB_PASS:
        return {"task_id": task_id, "status": "PASS", "error": "", "label": True}
    if stat == _BCB_TIMEOUT:
        return {"task_id": task_id, "status": "TIMEOUT", "error": "timeout", "label": False}
    if stat == _BCB_FAIL:
        return {"task_id": task_id, "status": "FAIL", "error": str(details)[:500], "label": False}
    return {"task_id": task_id, "status": "UNKNOWN", "error": str(stat), "label": False}


def evaluate_bigcodebench_one(out: InferenceOutputExample, bcb_repo: str | None = None) -> Dict[str, Any]:
    """
    Evaluate a single BigCodeBench output example.

    This helper is a thin wrapper around `_evaluate_one` so that programmatic
    callers (e.g. GEPA evolution) can reuse the exact same semantics as the
    CLI evaluation pipeline.
    """
    return _evaluate_one(out, bcb_repo=bcb_repo)


def evaluate_bigcodebench(
    output_file: str,
    bcb_repo: str | None = None,
    num_workers: int = 1,
) -> None:
    outputs = _load_outputs(output_file)
    if not outputs:
        print(f"[BigCodeBench Eval] No outputs found in {output_file}")
        return

    _ensure_bigcodebench_on_path(bcb_repo)

    evaluated: List[Dict[str, Any]] = []
    pass_count = 0
    subset = outputs[0].metadata.get("subset", "unknown")
    prompt_split = outputs[0].metadata.get("prompt_split", "unknown")

    if num_workers is None or num_workers < 1:
        num_workers = 1

    if num_workers == 1:
        for out in tqdm(outputs, total=len(outputs), desc="BigCodeBench eval", unit="task"):
            res = evaluate_bigcodebench_one(out, bcb_repo=bcb_repo)
            out_dict = out.to_dict()
            out_dict["label"] = bool(res["label"])
            out_dict["eval_status"] = res["status"]
            out_dict["eval_error"] = res["error"]
            evaluated.append(out_dict)
            pass_count += 1 if res["label"] else 0
    else:
        ordered: List[Dict[str, Any] | None] = [None] * len(outputs)

        def _run_one(idx: int, out: InferenceOutputExample):
            return idx, out, evaluate_bigcodebench_one(out, bcb_repo=bcb_repo)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_run_one, idx, out)
                for idx, out in enumerate(outputs)
            ]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="BigCodeBench eval",
                unit="task",
            ):
                idx, out, res = fut.result()
                out_dict = out.to_dict()
                out_dict["label"] = bool(res["label"])
                out_dict["eval_status"] = res["status"]
                out_dict["eval_error"] = res["error"]
                ordered[idx] = out_dict
                pass_count += 1 if res["label"] else 0

        evaluated = [item for item in ordered if item is not None]

    with open(output_file, "w", encoding="utf-8") as f:
        for item in evaluated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = len(evaluated)
    pass_at_1 = pass_count / total if total else 0.0
    print("\n" + "=" * 60)
    print("BigCodeBench Evaluation")
    print("=" * 60)
    print(f"Subset      : {subset}")
    print(f"Prompt split: {prompt_split}")
    print(f"Total       : {total}")
    print(f"Pass        : {pass_count}")
    print(f"Pass@1      : {pass_at_1:.4f}")
    print(f"Output file : {output_file}")
    print("=" * 60 + "\n")
