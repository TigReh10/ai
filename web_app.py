"""Minimal web chat UI for SmartAI (Flask).

Usage:
    python web_app.py
Then open http://localhost:5000
"""
from __future__ import annotations

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from agent import SmartAI

load_dotenv()
app = Flask(__name__)
ai = SmartAI()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"reply": "Please type a message."})
    reply = ai.ask(message, verbose=False)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
