import os
import uuid
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, Response
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


def stream_response(messages, conv_id, max_tokens=350):
    """Stream response from Anthropic API using SSE to avoid proxy timeouts."""
    def generate():
        full_text = ""
        with client.messages.stream(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                full_text += text
                yield f"data: {json.dumps({'delta': text})}\n\n"

        # Save full response to conversation history
        conversations[conv_id].append({"role": "assistant", "content": full_text})
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
    return stream_response(opening_msg, conv_id, max_tokens=200)


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

    return stream_response(messages, conv_id)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
