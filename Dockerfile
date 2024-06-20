FROM inseefrlab/onyxia-python-pytorch:py3.12.3-gpu

USER root

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir --upgrade -r requirements.txt && \
    chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT [ "./entrypoint.sh" ]
