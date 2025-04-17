#!/bin/bash

# Run linoprefs setup if the argument "linoprefs" is passed
if [[ "$1" == "linoprefs" ]]; then
  curl -sS https://raw.githubusercontent.com/linogaliana/init-scripts/refs/heads/main/install-copilot.sh | bash
fi

# Clone the repository and set up the environment
git clone https://github.com/InseeFrLab/llm-open-data-insee.git
cd llm-open-data-insee

curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -r pyproject.toml --system

pre-commit install
