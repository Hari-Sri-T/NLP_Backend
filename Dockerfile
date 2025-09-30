# Use an official Python image as a base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# The command to run your Gunicorn server, binding to the correct host and port
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--worker-tmp-dir", "/dev/shm", "--timeout", "120", "app:app"]