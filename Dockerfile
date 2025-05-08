FROM inseefrlab/onyxia-vscode-python:py3.12.9-2025.05.05

ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN git clone https://github.com/InseeFrLab/llm-open-data-insee.git --depth 1

WORKDIR /llm-open-data-insee

RUN pip install uv
RUN uv pip install -r pyproject.toml --system

EXPOSE 8000
CMD ["streamlit", "run", "app.py"]


