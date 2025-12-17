import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.nsfw import OpenNSFW2

# 1. Config
ModelPath = os.getenv("NSFW_MODEL_PATH", "models/opennsfw2.quant.onnx")
MaxImageSize = 15 * 1024 * 1024  # 15MB limit

# 2. Init Model
try:
    print(f"Loading model from {ModelPath}...")
    model = OpenNSFW2(ModelPath)
except Exception as e:
    print(f"Failed to init model: {e}")
    model = None

app = FastAPI(title="Moderation Inference Service")

class InferRequest(BaseModel):
    image_url: str

class InferResponse(BaseModel):
    nsfw_score: float
    raw: list[float]
    model: str
    latency_ms: float

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model failed to load")
    return {"ok": True}

@app.post("/infer/image", response_model=InferResponse)
def infer_image(req: InferRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    start_time = time.time()
    
    # 1. Fetch Image with size/timeout guard
    try:
        # Stream request to check headers first
        with requests.get(req.image_url, stream=True, timeout=15) as r:
            r.raise_for_status()
            
            # Content-Length check
            if 'Content-Length' in r.headers:
                if int(r.headers['Content-Length']) > MaxImageSize:
                    raise HTTPException(status_code=400, detail="Image too large (max 15MB)")
            
            # Read content (with hard limit in case header is missing/fake)
            content = b""
            for chunk in r.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > MaxImageSize:
                    raise HTTPException(status_code=400, detail="Image too large (max 15MB)")
                    
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Image fetch timed out")
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching image: {str(e)}")

    # 2. Inference
    try:
        result = model.predict(content)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        
        return {
            "nsfw_score": result["nsfw_score"],
            "raw": result["raw"],
            "model": os.path.basename(ModelPath),
            "latency_ms": round(latency, 2)
        }
        
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {str(e)}")
