import os
import tensorflow as tf
import tf2onnx
import opennsfw2 as n2

def export_to_onnx(output_path="opennsfw2.onnx"):
    print("Loading OpenNSFW2 model...")
    model = n2.make_open_nsfw_model()
    
    # Define input signature
    # Input shape is (Batch, 224, 224, 3)
    input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input')]
    
    print(f"Converting model to ONNX and saving to {output_path}...")
    # Convert to ONNX
    # opset 13 is a good default for modern runtimes
    tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13, output_path=output_path)
    
    print("Conversion complete.")

if __name__ == "__main__":
    export_to_onnx()
