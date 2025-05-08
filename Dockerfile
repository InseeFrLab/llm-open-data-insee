FROM inseefrlab/onyxia-vscode-python:py3.12.9-2025.05.05

ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN git clone https://github.com/InseeFrLab/llm-open-data-insee.git --depth 1

# Check what's in the top-level after cloning
RUN ls -al /llm-open-data-insee

# Optional: check whole cloned folder structure
RUN find /llm-open-data-insee

# Adjust to correct subdirectory (if needed)
WORKDIR /llm-open-data-insee/llm-open-data-insee

# Check that pyproject.toml is present
RUN ls -al

RUN pip install uv
RUN uv pip install -r pyproject.toml --system

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
