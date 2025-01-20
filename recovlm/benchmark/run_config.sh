# Install vLLM with CUDA 11.8.
#export VLLM_VERSION=0.6.1.post1
#export PYTHON_VERSION=310
#pip3 install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
#pip3 install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
apt-get install openjdk-8-jdk
pip3 install scikit-learn
pip3 install Levenshtein
pip install "git+https://github.com/salaniz/pycocoevalcap.git"
