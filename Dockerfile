FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8080

WORKDIR /app

COPY toolbridge/ /app/toolbridge/

EXPOSE 8080

CMD ["python", "-m", "toolbridge"]
