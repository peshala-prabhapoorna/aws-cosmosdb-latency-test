FROM python:3.13-slim

WORKDIR /app

RUN apt-get update

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["python3", "-m", "unittest", "test.py"]
