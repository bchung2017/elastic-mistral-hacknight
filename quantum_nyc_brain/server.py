"""Serve graph_ui.html and run its grammar against the real index.

pip install flask
export ELASTIC_ENDPOINT=... ELASTIC_API_KEY=... MISTRAL_API_KEY=...
python server.py            # http://127.0.0.1:5000  -> graph_ui.html, "Run on backend" posts to /query

POST /query  {"positive": "rat", "not": ["restaurant"], "any_of": []}  -> engine.search() output
"""
from pathlib import Path

from flask import Flask, jsonify, request, send_file

import engine

app = Flask(__name__)


@app.get("/")
def ui():
    return send_file(Path(__file__).with_name("graph_ui.html"))


@app.post("/query")
def query():
    body = request.get_json(force=True)
    p = engine.Parsed.model_validate({"positive": body.get("positive") or None,
                                      "not": body.get("not") or [], "any_of": body.get("any_of") or []})
    try:
        return jsonify(engine.search(p, int(body.get("size", 10))))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=5000, debug=False)
