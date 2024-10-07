FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg libegl1-mesa libegl1

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    git+https://github.com/XPixelGroup/BasicSR facexlib realesrgan lightning wandb torch-ema einops timm torch-fidelity \
    https://github.com/camenduru/wheels/releases/download/tost/natten-0.17.1-cp310-cp310-linux_x86_64.whl && \
    git clone https://github.com/camenduru/PMRF-hf /content/PMRF && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PMRF/raw/main/config.json -d /content/PMRF/model -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PMRF/resolve/main/model.safetensors -d /content/PMRF/model -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PMRF/resolve/main/RealESRGAN_x4plus.pth -d /content/PMRF/pretrained_models -o RealESRGAN_x4plus.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth -d /content/PMRF/pretrained_models -o RealESRGAN_x2plus.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth -d /home/camenduru/.cache/torch/hub/checkpoints -o weights-inception-2015-12-05-6726825d.pth

COPY ./worker_runpod.py /content/PMRF/worker_runpod.py
WORKDIR /content/PMRF
CMD python worker_runpod.py