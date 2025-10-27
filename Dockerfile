FROM ubuntu:22.04

RUN apt update && apt install -y curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /code/anomaly-detection-using-soundn
# CMD ["tail", "-f", "/dev/null"]
CMD ["/bin/sh", "-c", "bash"]

# CMD source ~/.bashrc
# CMD uv init -p 3.11
# CMD uv venv

# CMD uv pip install pandas
# CMD uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

