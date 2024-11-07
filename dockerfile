FROM python:3.8-slim
WORKDIR /app

# Install dependencies
RUN pip install numpy pandas scikit-learn jax jaxlib flax optax
 
# Copy the model and setup code
COPY . /app

# Set the entry point to start the training script
ENTRYPOINT ["python", "training_script.py"]
