/**
 * Cloudflare Worker to orchestrate NSFW moderation.
 * - Receives request with image URL.
 * - Calls the containerized inference service.
 * - Returns the moderation result.
 */

export default {
    async fetch(request, env, ctx) {
        // Only accept POST
        if (request.method !== "POST") {
            return new Response("Method not allowed", { status: 405 });
        }

        try {
            const body = await request.json();
            const imageUrl = body.image_url;

            if (!imageUrl) {
                return new Response("Missing image_url", { status: 400 });
            }

            if (!env.CONTAINER_URL) {
                return new Response("Configuration Error: CONTAINER_URL not set", { status: 500 });
            }

            // Call the Container Inference Service
            // If you secured your container with an API key, include it here.
            // For Cloudflare Containers (beta), mTLS or Service Tokens might be preferred,
            // but assuming simple header auth for now if configured.
            const containerResponse = await fetch(`${env.CONTAINER_URL}/infer/image`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    // "X-API-Key": env.API_KEY || "" // Uncomment if using API key
                },
                body: JSON.stringify({ image_url: imageUrl }),
            });

            if (!containerResponse.ok) {
                const errorText = await containerResponse.text();
                return new Response(`Inference Service Error: ${errorText}`, {
                    status: containerResponse.status
                });
            }

            const result = await containerResponse.json();
            const score = result.nsfw_score;

            // Decision Logic (Based on your table)
            let action = "ALLOW";
            let blur = false;
            let review = false;

            if (score < 0.2) {
                // Safe
                action = "ALLOW";
            } else if (score < 0.6) {
                // Uncertain - Monitor
                action = "ALLOW_MONITOR";
            } else if (score < 0.85) {
                // Sensitive - Blur
                action = "BLUR";
                blur = true;
            } else {
                // Likely NSFW (> 0.85) - Blur + Review
                // No auto-blocking; human review required.
                action = "FLAG";
                blur = true;
                review = true;
            }

            // Return the result + decision to your backend/client
            return new Response(JSON.stringify({
                ...result,
                decision: { action, blur, review }
            }), {
                headers: { "Content-Type": "application/json" },
            });

        } catch (e) {
            return new Response(`Worker Error: ${e.message}`, { status: 500 });
        }
    },
};
