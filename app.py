import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from a .env file in the same folder

app = Flask(__name__, static_folder=".", static_url_path="")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # change to any model your Groq account supports

SYSTEM_PROMPT = "You are LISAX, a friendly, concise, and helpful AI chat assistant."


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY is not set on the server."}), 500

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history") or []  # list of {role, content}

    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        reply = result["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})
    except requests.exceptions.HTTPError:
        return jsonify({"error": "Groq API error", "details": resp.text}), resp.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to reach Groq API", "details": str(e)}), 502


if __name__ == "__main__":
    app.run(debug=True, port=5000)