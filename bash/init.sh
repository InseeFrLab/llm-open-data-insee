#!/bin/bash

sudo apt update -y
sudo apt install tree -y

# Replace default flake8 linter with project-preconfigured ruff
code-server --uninstall-extension ms-python.flake8
code-server --install-extension charliermarsh.ruff

# Install type checking extension
code-server --install-extension ms-python.mypy-type-checker

# Dev extensions
code-server --install-extension oderwat.indent-rainbow
code-server --install-extension pomdtr.excalidraw-editor


# Define the configuration directory for VS Code
VSCODE_CONFIG_DIR="$HOME/.local/share/code-server/User"

# Create the configuration directory if necessary
mkdir -p "$VSCODE_CONFIG_DIR"

# User settings file
SETTINGS_FILE="$VSCODE_CONFIG_DIR/settings.json"

# Enable dark mode by default
echo '{
    "workbench.colorTheme": "Default Dark Modern"
}' > "$SETTINGS_FILE"

# Keybindings file
KEYBINDINGS_FILE="$VSCODE_CONFIG_DIR/keybindings.json"

# Add shortcuts for duplicating and deleting lines
echo '[
    {
        "key": "ctrl+shift+d",
        "command": "editor.action.duplicateSelection"
    },
    {
        "key": "ctrl+d",
        "command": "editor.action.deleteLines",
        "when": "editorTextFocus"
    }
]' > "$KEYBINDINGS_FILE"


cd ~/work/
git clone https://github.com/InseeFrLab/llm-open-data-insee.git
cd llm-open-data-insee

# Install requirements and run linting on project
pip install -r requirements-dev.txt
mypy --install-types
pre-commit install
pre-commit run --all-files

# Run nbstripout installation command in the terminal
echo "Running nbstripout --install..."
nbstripout --install



# mc cp -r s3/projet-llm-insee-open-data/data/chroma_database/chroma_db/ ~/work/data/chroma_db
