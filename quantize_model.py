import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(input_model_path="opennsfw2.onnx", output_model_path="opennsfw2.quant.onnx"):
    print(f"Quantizing model {input_model_path} to {output_model_path}...")
    
    # Quantize the model
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8  # Quantize weights to uint8
    )
    
    print("Quantization complete.")
    
    # Verify file sizes
    original_size = os.path.getsize(input_model_path)
    quantized_size = os.path.getsize(output_model_path)
    
    print(f"Original model size: {original_size / (1024*1024):.2f} MB")
    print(f"Quantized model size: {quantized_size / (1024*1024):.2f} MB")
    print(f"Reduction: {original_size / quantized_size:.2f}x")

if __name__ == "__main__":
    import os
    quantize_onnx_model()
