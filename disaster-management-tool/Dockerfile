FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
