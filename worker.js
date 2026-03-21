/**
 * EmbedClipFarm — Cloudflare Worker Proxy
 *
 * Fetches YouTube pages and transcript XML with proper headers
 * to bypass bot detection. Deploy this to Cloudflare Workers (free tier).
 *
 * Setup:
 * 1. Go to https://workers.cloudflare.com → sign up / log in
 * 2. Create a new Worker
 * 3. Paste this entire file → Deploy
 * 4. Copy your worker URL (e.g. https://my-worker.username.workers.dev)
 * 5. Enter it in EmbedClipFarm settings
 */

export default {
  async fetch(request) {
    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }

    const url = new URL(request.url);
    const targetUrl = url.searchParams.get("url");

    if (!targetUrl) {
      return jsonResponse({ error: "Missing ?url= parameter" }, 400);
    }

    // Only allow YouTube domains
    let parsed;
    try {
      parsed = new URL(targetUrl);
    } catch {
      return jsonResponse({ error: "Invalid URL" }, 400);
    }

    if (!parsed.hostname.endsWith("youtube.com") && !parsed.hostname.endsWith("youtu.be")) {
      return jsonResponse({ error: "Only YouTube URLs allowed" }, 403);
    }

    try {
      const resp = await fetch(targetUrl, {
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
          "Accept-Language": "en-US,en;q=0.9",
          "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        redirect: "follow",
      });

      const body = await resp.text();

      return new Response(body, {
        status: resp.status,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Content-Type": resp.headers.get("Content-Type") || "text/html",
        },
      });
    } catch (err) {
      return jsonResponse({ error: err.message }, 500);
    }
  },
};

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Content-Type": "application/json",
    },
  });
}
