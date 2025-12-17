import io
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

# Config for output ordering
# OpenNSFW2 usually outputs [SFW, NSFW] -> index 1 is NSFW
# But we allow override via env var just in case
NSFW_INDEX = int(os.getenv("NSFW_INDEX", "1"))

class OpenNSFW2:
    def __init__(self, model_path: str):
        # CPUExecutionProvider for Cloudflare Containers (CPU)
        try:
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = self.sess.get_outputs()[0].name
            print(f"Model loaded. NSFW Index configured to: {NSFW_INDEX}")
        except Exception as e:
            print(f"CRITICAL: Failed to load ONNX model: {e}")
            raise e

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocessing for OpenNSFW2.
        CRITICAL: This model was trained with 'Yahoo' style preprocessing.
        - Resize to 256x256
        - Center crop 224x224
        - Convert to BGR
        - Subtract VGG mean [104, 117, 123]
        - DO NOT normalize to 0-1 or use ImageNet mean/std.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 1. Resize to (256, 256)
        img = img.resize((256, 256), resample=Image.BILINEAR)
        
        # 2. Center crop (224, 224)
        width, height = img.size
        target_size = 224
        
        left = (width - target_size) / 2
        top = (height - target_size) / 2
        right = (width + target_size) / 2
        bottom = (height + target_size) / 2
        
        img = img.crop((left, top, right, bottom))
        
        # Convert to numpy array (H, W, C)
        arr = np.asarray(img, dtype=np.float32)
        
        # 3. RGB to BGR
        arr = arr[:, :, ::-1]
        
        # 4. Subtract mean [104, 117, 123] (VGG Mean)
        vgg_mean = np.array([104, 117, 123], dtype=np.float32)
        arr = arr - vgg_mean
        
        # 5. Expand dims to (1, 224, 224, 3) - NHWC
        # Note: Some ONNX exports might be NCHW. If the model fails, check input shape.
        # Based on typical TF->ONNX conversion, it retains NHWC unless explicitly changed.
        arr = np.expand_dims(arr, axis=0)
        
        return arr

    def predict(self, image_bytes: bytes) -> dict:
        x = self._preprocess(image_bytes)
        
        # Run inference
        y = self.sess.run([self.output_name], {self.input_name: x})[0]
        
        scores = y[0].tolist()
        
        # Safety check for index
        if NSFW_INDEX < 0 or NSFW_INDEX >= len(scores):
             raise ValueError(f"NSFW_INDEX {NSFW_INDEX} is out of bounds for output size {len(scores)}")

        return {
            "nsfw_score": scores[NSFW_INDEX],
            "raw": scores
        }
