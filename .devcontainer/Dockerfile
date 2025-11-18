FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt install -y \
    # python build
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libbz2-dev liblzma-dev libsqlite3-dev \
    # clangd 
    wget curl lsb-release software-properties-common gnupg \
    # etc...
    wget git sudo curl cmake \
    # serial
    libboost-dev \
    # Qt6
    qt6-base-dev qt6-tools-dev libgl1-mesa-dev

# Tkinter backend
RUN apt install -y tk-dev libx11-dev libxext-dev libxrender-dev libxft-dev
# PyQt6 backend
RUN apt install -y libx11-xcb1 libxkbcommon-x11-0 libxcb1 libxcb-shm0 libxcb-xfixes0 \
    libxcb-render0 libxcb-render-util0 libgl1 libglu1-mesa libxi6 libsm6 \
    libxrandr2 libxtst6 libfontconfig1 libfreetype6 
# PyQt6 runtime
RUN apt install -y libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-shape0 libxcb-sync1 libxcb-xinerama0 libxcb-xinput0 libegl1 libglvnd0 libdbus-1-3

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
    && mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2204-13-0-local_13.0.2-580.95.05-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu2204-13-0-local_13.0.2-580.95.05-1_amd64.deb \
    && cp /var/cuda-repo-ubuntu2204-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get -y install cuda-toolkit-13-0

RUN wget https://developer.download.nvidia.com/compute/cudnn/9.16.0/local_installers/cudnn-local-repo-ubuntu2204-9.16.0_1.0-1_amd64.deb \
    && dpkg -i cudnn-local-repo-ubuntu2204-9.16.0_1.0-1_amd64.deb \
    && cp /var/cudnn-local-repo-ubuntu2204-9.16.0/cudnn-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get -y install cudnn

RUN wget https://developer.download.nvidia.com/compute/tensorrt/10.14.1/local_installers/nv-tensorrt-local-repo-ubuntu2204-10.14.1-cuda-13.0_1.0-1_amd64.deb \
    && dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.14.1-cuda-13.0_1.0-1_amd64.deb \
    && cp /var/nv-tensorrt-local-repo-ubuntu2204-10.14.1-cuda-13.0/nv-tensorrt-local-DC5C259A-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get -y install tensorrt

# for nvcc
ENV PATH=${PATH}:/usr/local/cuda-13.0/bin
ENV PATH=${PATH}:/usr/src/tensorrt/bin
# # for tensorRT
ENV TENSORRT_DIR=/usr/local/tensorrt
ENV LD_LIBRARY_PATH=$TENSORRT_DIR/lib
ENV CPLUS_INCLUDE_PATH=$TENSORRT_DIR/include


# Install cudatoolkit, cudnn (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-12x)
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    # dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    # apt-get update && apt-get -y install cuda-toolkit-12-9 cudnn9-cuda-12

# TensorRT
# RUN wget https://developer.download.nvidia.com/compute/tensorrt/10.14.1/local_installers/nv-tensorrt-local-repo-ubuntu2204-10.14.1-cuda-12.9_1.0-1_amd64.deb

# setting clangd 21 version(https://apt.llvm.org/)
RUN wget https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh \
    && chmod +x /tmp/llvm.sh \ 
    && /tmp/llvm.sh 21 \
    && ln -s /usr/bin/clangd-21 /usr/bin/clangd 

# Build and install python 3.12.12 version
RUN cd /tmp \
    && wget https://www.python.org/ftp/python/3.12.12/Python-3.12.12.tgz \
    && tar -xf Python-3.12.12.tgz \
    && cd Python-3.12.12 \
    && ./configure --enable-optimizations \
    && make -j $(nproc) \
    && make install \
    && rm -rf /tmp/*

# Apply from python3.12 to python or python3
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1 \
    && python -m pip install --no-cache-dir --upgrade pip

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN pip install pyqt6 pyside6 matplotlib pandas pyserial pyqtgraph scipy

# Install the cuda toolkit (https://developer.nvidia.com/cuda-toolkit-archive)
# RUN wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run \
    # && sudo sh cuda_12.9.1_575.57.08_linux.run
# ONNX-runtime (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
# RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz \
    # && tar -xzf 

# Using non-root system
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USERNAME -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
    
USER $USERNAME

# etc...
SHELL ["/bin/bash", "-c"]

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash \
    && source ~/.nvm/nvm.sh \
    && nvm install 24 \
    && npm install -g @google/gemini-cli@latest

# Make venv environment
# RUN cd /workspaces/anomaly-detection-using-sound \
    # && python3.12.12 -m venv anomaly-detection


# RUN curl -LsSf https://astral.sh/uv/install.sh | sh


# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USERNAME -m -s /bin/bash $USERNAME \
#     && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME


CMD [ "/bin/bash" ]