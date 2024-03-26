# Obtain and start the basic docker image environment.

docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install dependencies, TensorRT-LLM requires Python 3.10

apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev

# Install the latest preview version (corresponding to the main branch) of TensorRT-LLM.

# If you want to install the stable version (corresponding to the release branch), please

# remove the `--pre` option.

pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

pip install mpmath==1.3.0

# Check installation

python3 -c "import tensorrt_llm"

# You need Git LFS to run

# if not installed, do the following before moving to next step

apt-get install sudo
apt-get install git
sudo apt-get install git-lfs

# Next Step: If installed LFS do the following

# Problem: This took me quite a lot of time, optimize build?

cd home/
git clone https://github.com/NVIDIA/TensorRT-LLM.git
pip install -r examples/mixtral/requirements.txt
git lfs install

python benchmark.py \
 -m mixtral_7b \
 --mode plugin \
 --batch_size "1;8;64" \
 --input_output_len "60,20;128,20"

http://localhost:8002/throughput_vs_latency.png

# Build Triton

# Prepare the TRT-LLM base image using the dockerfile from tensorrtllm_backend.

git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend

# Specify the build args for the dockerfile.

BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.01-py3-min
TRT_VERSION=9.2.0.5
TRT_URL_x86=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz
TRT_URL_ARM=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.Ubuntu-22.04.aarch64-gnu.cuda-12.2.tar.gz

docker build -t trtllm_base \
 --build-arg BASE_IMAGE="${BASE_IMAGE}" \
             --build-arg TRT_VER="${TRT_VERSION}" \
 --build-arg RELEASE_URL_TRT_x86="${TRT_URL_x86}" \
             --build-arg RELEASE_URL_TRT_ARM="${TRT_URL_ARM}" \
 -f dockerfile/Dockerfile.triton.trt_llm_backend .

# Run the build script from Triton Server repo. The flags for some features or

# endpoints can be removed if not needed. Please refer to the support matrix to

# see the aligned versions: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html

TRTLLM_BASE_IMAGE=trtllm_base
TENSORRTLLM_BACKEND_REPO_TAG=v0.7.2
PYTHON_BACKEND_REPO_TAG=r24.01
git clone https://github.com/triton-inference-server/server.git
cd server
./build.py -v --no-container-interactive --enable-logging --enable-stats --enable-tracing \
 --enable-metrics --enable-gpu-metrics --enable-cpu-metrics \
 --filesystem=gcs --filesystem=s3 --filesystem=azure_storage \
 --endpoint=http --endpoint=grpc --endpoint=sagemaker --endpoint=vertex-ai \
 --backend=ensemble --enable-gpu --endpoint=http --endpoint=grpc \
 --no-container-pull \
 --image=base,${TRTLLM_BASE_IMAGE} \
              --backend=tensorrtllm:${TENSORRTLLM_BACKEND_REPO_TAG} \
 --backend=python:${PYTHON_BACKEND_REPO_TAG}

Host TensorRT LLM

# Update the submodule TensorRT-LLM repository

git submodule update --init --recursive
git lfs install
git lfs pull

# TensorRT-LLM is required for generating engines. You can skip this step if

# you already have the package installed. If you are generating engines within

# the Triton container, you have to install the TRT-LLM package.

(cd tensorrt_llm &&
bash docker/common/install_cmake.sh &&
export PATH=/usr/local/cmake/bin:$PATH &&
python3 ./scripts/build_wheel.py --trt_root="/usr/local/tensorrt" &&
pip3 install ./build/tensorrt_llm\*.whl)

# Go to the tensorrt_llm/examples/gpt directory

cd tensorrt_llm/examples/mixtral

# Download weights from HuggingFace Transformers

git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1

# Convert weights from HF Tranformers to TensorRT-LLM checkpoint

python convert_checkpoint.py --model_dir ./Mixtral-8x7B-v0.1 \
 --output_dir ./tllm_checkpoint_mixtral_2gpu \
 --dtype float16 \
 --pp_size 2

# Build TensorRT engines

trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_2gpu \
 --output_dir ./trt_engines/mixtral/pp2 \
 --gemm_plugin float16

docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 bash
