"""
Local proxy server — forwards court site requests from Colab through your laptop.
Colab gets blocked (datacenter IP), your laptop doesn't (home IP).

Run on your laptop:
    python scraper/proxy_server.py

Then paste the ngrok URL into Colab.
"""

import requests
from flask import Flask, request, Response
from pyngrok import ngrok

app = Flask(__name__)

USER_AGENT = (
    "MSDS603-Research-Scraper/1.0 "
    "(SF Small Claims academic study; contact: msds603-team@usfca.edu)"
)


@app.route("/proxy")
def proxy():
    url = request.args.get("url")
    if not url:
        return {"error": "missing url param"}, 400

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
            stream=True,
        )
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("content-type", "application/octet-stream"),
        )
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    tunnel = ngrok.connect(5000)
    print(f"\n{'='*50}")
    print(f"PROXY URL (paste into Colab):")
    print(f"  {tunnel.public_url}")
    print(f"{'='*50}\n")
    app.run(port=5000)
