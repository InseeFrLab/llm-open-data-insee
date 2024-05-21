FROM inseefrlab/onyxia-vscode-pytorch:py3.12.2-gpu

USER root

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir --upgrade -r requirements-app.txt && \
    chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT [ "./entrypoint.sh" ]
