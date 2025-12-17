import onnxruntime as ort
import numpy as np
import opennsfw2 as n2
from PIL import Image
import time

def run_inference(model_path="opennsfw2.quant.onnx"):
    print(f"Loading model from {model_path}...")
    session = ort.InferenceSession(model_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Create a dummy image (or load one if available)
    # Using a random image for demonstration
    print("Creating dummy input...")
    # OpenNSFW2 expects (224, 224, 3) input with specific preprocessing
    # We can use n2.preprocess_image if we have a PIL image, or manually create numpy array
    # Let's create a random numpy array to simulate preprocessed input
    # Preprocessing: resize to 256x256, crop to 224x224, subtract mean, etc.
    # Here we just pass random data to verify the model runs
    
    # Input shape: (Batch, 224, 224, 3)
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    print("Running inference...")
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})
    end_time = time.time()
    
    print("Inference successful!")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output probabilities: {outputs[0]}")
    print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")

if __name__ == "__main__":
    run_inference()
