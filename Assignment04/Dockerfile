# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and templates
COPY . /app

# Expose port used by Flask
EXPOSE 5000

ENV FLASK_APP=app.py
ENV PORT=5000

CMD ["python", "app.py"]