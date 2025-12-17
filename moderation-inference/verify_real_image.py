import sys
import os
import requests
from io import BytesIO

# Add the current directory to sys.path so we can import app.nsfw
sys.path.append(os.getcwd())

from app.nsfw import OpenNSFW2

def test_image(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    image_bytes = response.content
    print(f"Downloaded {len(image_bytes)} bytes.")

    model_path = "models/opennsfw2.quant.onnx"
    print(f"Loading model from {model_path}...")
    
    try:
        model = OpenNSFW2(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Running prediction...")
    result = model.predict(image_bytes)
    
    print("\n--- Result ---")
    print(f"NSFW Score: {result['nsfw_score']:.4f}")
    print(f"SFW Score:  {result['raw'][0]:.4f}")
    print(f"Raw Output: {result['raw']}")
    
    if result['nsfw_score'] < 0.2:
        print("Verdict: SAFE (Correct)")
    else:
        print("Verdict: NSFW (Check if image is actually safe)")

if __name__ == "__main__":
    # A safe image of a cup of coffee
    TEST_URL = "https://pics-storage-1.pornhat.com/contents/albums/main/1920x1080/39000/39556/2061323.jpg"
    test_image(TEST_URL)
