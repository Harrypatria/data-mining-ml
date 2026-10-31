FROM python:3.11
COPY . /app
RUN pip install -r /app/requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--reload"]
