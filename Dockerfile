FROM inseefrlab/onyxia-vscode-pytorch:py3.12.2-gpu

USER root

WORKDIR /api

COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade -r requirements-api.txt

COPY ./src /api/src

EXPOSE 8000
CMD ["uvicorn", "src.api:app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
