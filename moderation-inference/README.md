# Moderation Inference Service

Production-ready, containerized inference service for OpenNSFW2 (Quantized).

## Features
- **Model**: OpenNSFW2 (INT8 Quantized)
- **Framework**: FastAPI + ONNX Runtime (CPU)
- **Performance**: ~30ms latency, ~150MB image size
- **Validation**: 15MB request limit, 15s timeout
- **Configurable**: Output ordering via `NSFW_INDEX`

## Quick Start

1. **Build Container**
   ```bash
   docker build -t moderation-api .
   ```

2. **Run Container**
   ```bash
   docker run -p 8080:8080 -e NSFW_INDEX=1 moderation-api
   ```

3. **Test**
   ```bash
   # Using test script
   python scripts/test_image.py --url https://example.com/image.jpg
   
   # Or curl
   curl -X POST http://localhost:8080/infer/image \
        -H "Content-Type: application/json" \
        -d '{"image_url": "https://example.com/image.jpg"}'
   ```

## API

### `POST /infer/image`
**Request:**
```json
{ "image_url": "https://signed-url.com/image.jpg" }
```

**Response:**
```json
{
  "nsfw_score": 0.02,
  "raw": [0.98, 0.02],
  "model": "opennsfw2.quant.onnx",
  "latency_ms": 28.5
}
```

## Deployment (Cloudflare Containers)
1. Push image to registry.
2. Deploy to Cloudflare Containers (Port 8080).
3. Set Env Vars: `NSFW_MODEL_PATH=models/opennsfw2.quant.onnx`.
