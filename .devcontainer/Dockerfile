FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    # python build
    build-essential \
    wget \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    libsqlite3-dev \
    # Tkinter backend
    tk-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxft-dev \
    # PyQt6 backend
    libx11-xcb1 \
    libxkbcommon-x11-0 \
    libxcb1 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libgl1 \
    libglu1-mesa \
    libxi6 \
    libsm6 \
    libxrandr2 \
    libxtst6 \
    libfontconfig1 \
    libfreetype6 \
    # PyQt6 runtime
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libegl1 \
    libglvnd0 \
    libdbus-1-3 \
    # etc
    git \
    sudo \
    curl

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
    &&python -m pip install --no-cache-dir --upgrade pip

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN pip install pyqt6 pyside6 matplotlib pandas pyserial pyqtgraph scipy


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