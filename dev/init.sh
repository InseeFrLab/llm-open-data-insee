#!/bin/bash

git clone https://github.com/InseeFrLab/llm-open-data-insee.git
cd llm-open-data-insee

curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -r pyproject.toml --system

pre-commit