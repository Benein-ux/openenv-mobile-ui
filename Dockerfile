# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY server/ ./server/

# Create a non-root user for security (HF Spaces recommends this)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Expose the default Hugging Face Spaces port
EXPOSE 7860

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
