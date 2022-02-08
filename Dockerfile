# syntax=docker/dockerfile:1

FROM python:3.7.9

WORKDIR /tfrecorder

ENV VIRTUAL_ENV=/opt/env
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUALENV/bin:$PATH"
ENV PYTHONPATH="$PYTHONPATH:/tfrecorder"

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["make", "test-py"]