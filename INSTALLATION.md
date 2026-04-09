```
conda create -n src-all python=3.11 -y
conda activate src-all
python -m pip install -U pip setuptools wheel

sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3-dev git

python -m pip install "numpy==1.26.4" "gym==0.23.1" "gymnasium==0.29.1"

python -m pip install alfworld==0.3.5 scienceworld==1.2.2 jericho minigrid jsonlines nltk wikipedia langchain pyyaml scipy matplotlib imageio

conda install -c conda-forge openjdk=11
java -version

python -m pip install -r data/bigcodebench/bigcodebench-main/Requirements/requirements.txt
python -m pip install --force-reinstall "opencv-python==4.9.0.80" "opencv-python-headless==4.9.0.80"  "numpy==1.26.4"

pip install gepa

pip install litellm

pip install scikit-image

python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

# FlashOAgents dependencies (required by evolution_memevolve PromptAnalyzer)
python -m pip install openai huggingface_hub pillow pyyaml jinja2 rich json-repair

# Optional but recommended for imports in remote shells:
# export PYTHONPATH="$PWD/MemEvolve/Flash-Searcher-main:$PYTHONPATH"
```
