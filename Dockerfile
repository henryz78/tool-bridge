FROM python:3.12-slim

WORKDIR /app

COPY toolbridge/ /app/toolbridge/

EXPOSE 8080

CMD ["python", "-m", "toolbridge"]
