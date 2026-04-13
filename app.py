import os
import uuid
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-fallback-key-change-me")

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text()

# Server-side conversation storage (keyed by session ID)
conversations = {}


@app.route("/")
def index():
    conv_id = str(uuid.uuid4())
    session["conv_id"] = conv_id
    conversations[conv_id] = []
    return render_template("index.html")


def get_response(messages, conv_id, max_tokens=350):
    """Get complete response from Anthropic API."""
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        full_text = message.content[0].text

        # Save full response to conversation history
        conversations[conv_id].append({"role": "assistant", "content": full_text})

        return jsonify({"message": full_text})

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/opening", methods=["POST"])
def opening():
    """Get Ray's opening message to start the conversation."""
    conv_id = session.get("conv_id")
    if not conv_id or conv_id not in conversations:
        conv_id = str(uuid.uuid4())
        session["conv_id"] = conv_id
        conversations[conv_id] = []

    opening_msg = [{"role": "user", "content": "[New conversation started. Send your opening message as Ray.]"}]
    conversations[conv_id] = list(opening_msg)
    return get_response(opening_msg, conv_id, max_tokens=200)


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

    return get_response(messages, conv_id)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
