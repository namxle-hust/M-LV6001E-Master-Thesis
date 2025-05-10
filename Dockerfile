FROM python:3.12.10-slim

WORKDIR /apps

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /workspace