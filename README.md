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
