// Cloudflare Worker Snippet

export default {
    async fetch(request, env) {
        if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });

        const { image_url } = await request.json();
        if (!image_url) return new Response("Missing image_url", { status: 400 });

        // 1. (Optional) Validate signed R2 URL here

        // 2. Call Inference Service
        const inferenceResponse = await fetch(`${env.CONTAINER_URL}/infer/image`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_url })
        });

        if (!inferenceResponse.ok) {
            return new Response(`Inference Error: ${inferenceResponse.statusText}`, { status: inferenceResponse.status });
        }

        const result = await inferenceResponse.json();

        // 3. Parse and Act
        // result format: { nsfw_score: 0.1, raw: [...], model: "...", latency_ms: 30 }

        return new Response(JSON.stringify(result), { headers: { "Content-Type": "application/json" } });
    }
};
