FROM inseefrlab/onyxia-vscode-pytorch:py3.12.2-gpu

USER root

WORKDIR /app

COPY requirements-app.txt .

RUN pip install --no-cache-dir --upgrade -r requirements-app.txt

COPY . /app/

EXPOSE 8000
ENTRYPOINT [ "entrypoint.sh" ]
