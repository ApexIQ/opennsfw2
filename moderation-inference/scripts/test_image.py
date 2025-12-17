import argparse
import requests
import json
import sys

def test_inference(url, api_url="http://localhost:8080/infer/image"):
    print(f"Testing image: {url}")
    print(f"Target API: {api_url}")
    
    payload = {"image_url": url}
    
    try:
        response = requests.post(api_url, json=payload, timeout=20)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\nResponse:")
            print(json.dumps(data, indent=2))
        else:
            print(f"\nError: {response.text}")
            
    except Exception as e:
        print(f"\nRequest Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Moderation API")
    parser.add_argument("--url", required=True, help="Image URL to test")
    parser.add_argument("--endpoint", default="http://localhost:8080/infer/image", help="API Endpoint")
    
    args = parser.parse_args()
    test_inference(args.url, args.endpoint)
