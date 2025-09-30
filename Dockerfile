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

# --- ADD THIS LINE TO DEBUG ---
# This will test if app.py can be imported. If it fails, the build logs will show the real error.
RUN python -c "import app"

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# The command to run your Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--worker-tmp-dir", "/dev/shm", "--timeout", "120", "app:app"]