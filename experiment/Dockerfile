# 1. Base image: Python + Linux
FROM python:3.11-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy dependency list first (for caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the experiment code
COPY . .

# 6. Default command to run the experiment
CMD ["python", "train.py"]

