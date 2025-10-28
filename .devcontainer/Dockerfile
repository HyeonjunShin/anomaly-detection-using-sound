FROM ubuntu:22.04

RUN apt update && apt install -y \
    curl \
    git

RUN curl -LsSf https://astral.sh/uv/install.sh | sh


# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USERNAME -m -s /bin/bash $USERNAME \
#     && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME



# CMD [ "/bin/bash" ]