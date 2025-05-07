FROM ubuntu:22.04

# Set noninteractive frontend to avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN git clone https://github.com/InseeFrLab/llm-open-data-insee.git --depth 1

WORKDIR /llm-open-data-insee

RUN pip install uv
RUN uv pip install -r pyproject.toml --system

EXPOSE 8000
CMD ["streamlit", "run", "app.py"]


