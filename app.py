import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-fallback-key-change-me")

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text()

# Server-side conversation storage (keyed by session ID)
conversations = {}


@app.route("/")
def index():
    conv_id = str(uuid.uuid4())
    session["conv_id"] = conv_id
    conversations[conv_id] = []
    return render_template("index.html")


@app.route("/api/opening", methods=["POST"])
def opening():
    """Get Ray's opening message to start the conversation."""
    conv_id = session.get("conv_id")
    if not conv_id or conv_id not in conversations:
        # Conversation lost (e.g. worker mismatch) — create a new one
        conv_id = str(uuid.uuid4())
        session["conv_id"] = conv_id
        conversations[conv_id] = []

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "[New conversation started. Send your opening message as Ray.]"}],
    )
    ray_text = response.content[0].text
    conversations[conv_id] = [
        {"role": "user", "content": "[New conversation started. Send your opening message as Ray.]"},
        {"role": "assistant", "content": ray_text},
    ]
    return jsonify({"response": ray_text})


@app.route("/api/chat", methods=["POST"])
def chat():
    conv_id = session.get("conv_id")
    if not conv_id or conv_id not in conversations:
        return jsonify({"error": "Session expired. Please refresh the page."}), 400

    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    messages = conversations[conv_id]
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=MODEL,
        max_tokens=350,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    ray_text = response.content[0].text
    messages.append({"role": "assistant", "content": ray_text})

    return jsonify({"response": ray_text})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
