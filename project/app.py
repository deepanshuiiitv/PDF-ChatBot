import os
from uuid import uuid4
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template
from rag_engine import process_pdf, ask_question
from pinecone_utils import clear_index

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/status')
def status():
    try:
        import rag_engine
        docs = len(getattr(rag_engine, 'DOCUMENTS', []))
        has_embed = rag_engine.EMBEDDINGS is not None
        pine = bool(os.getenv('PINECONE_API_KEY') and os.getenv('PINECONE_INDEX'))
        return jsonify({"documents": docs, "has_local_embeddings": has_embed, "pinecone_configured": pine})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_pdf():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    saved_paths = []
    for file in files:
        if not file or not file.filename.lower().endswith(".pdf"):
            continue
        filename = f"{uuid4().hex}_{secure_filename(file.filename)}"
        path = os.path.join(UPLOAD_DIR, filename)
        file.save(path)
        saved_paths.append(path)

    if not saved_paths:
        return jsonify({"error": "No valid PDF files uploaded"}), 400

    for path in saved_paths:
        try:
            process_pdf(path)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    session_id = uuid4().hex
    return jsonify({"status": "PDF(s) processed", "session_id": session_id})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    msg = data.get("message")
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    try:
        answer = ask_question(msg)
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/abort", methods=["POST"])
def abort_chat():
    data = request.get_json() or {}
    delete_index = bool(data.get("delete_index"))
    result = clear_index(delete_index=delete_index)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

