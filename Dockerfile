FROM inseefrlab/onyxia-vscode-python:py3.12.9-2025.05.05

ENV DEBIAN_FRONTEND=noninteractive

USER root

# Create working directory
WORKDIR /app

# Copy necessary files and directories
COPY app.py .
COPY pyproject.toml .
COPY src/ src/
COPY prompt/ prompt/

# Install uv and dependencies
RUN pip install uv
RUN uv pip install -r pyproject.toml --system

# Expose default Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
