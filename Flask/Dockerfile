# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the Docker requirements file
COPY docker-requirements.txt .

# Install basic dependencies
RUN pip install --no-cache-dir -r docker-requirements.txt

# Copy the application code
COPY . .

# Make port 5001 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]