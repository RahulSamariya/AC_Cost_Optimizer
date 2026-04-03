# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY setup.py .
# We don't have a requirements.txt, but we have setup.py. 
# Let's create a requirements.txt for simplicity in Docker.
RUN pip install --no-cache-dir numpy gymnasium pytest fastapi uvicorn pydantic requests

# Copy the rest of the application code
COPY . .

# Expose the required port
EXPOSE 7860

# Start the application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
