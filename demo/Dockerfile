# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8
RUN apt-get update &&\
    apt-get install -y gcc

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
ADD ./demo/requirements.txt .
RUN python -m pip install -r requirements.txt


COPY ./assets/ /assets/
COPY ./demo/ /demo/

WORKDIR /app
COPY ./src/ /app


# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd marcalph && chown -R marcalph /app
RUN mkdir /volume && chown -R marcalph /volume
USER marcalph

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
