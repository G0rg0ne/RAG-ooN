# Step 1: Choose the base image
FROM python:3.10-slim

# Step 2: Set environment variables to avoid Python buffering (good practice for logging)
ENV PYTHONUNBUFFERED=1

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Copy the requirements file
COPY requirements.txt .

# Step 5: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

