from flask import Flask, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the KYC Verification Service",
        "status": "running"
    })

@app.route("/verify", methods=["POST"])
def verify_wallet():
    try:
        id_front = request.files.get("id_front")
        id_back = request.files.get("id_back")
        selfie = request.files.get("selfie")

        if not id_front or not id_back or not selfie:
            return jsonify({"message": "Missing one or more files", "status": "error"}), 400

        # Save files
        id_front.save(os.path.join(UPLOAD_FOLDER, f"id_front_{id_front.filename}"))
        id_back.save(os.path.join(UPLOAD_FOLDER, f"id_back_{id_back.filename}"))
        selfie.save(os.path.join(UPLOAD_FOLDER, f"selfie_{selfie.filename}"))

        return jsonify({
            "message": "Wallet verification started!",
            "status": "pending",
            "files_received": [id_front.filename, id_back.filename, selfie.filename]
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({"message": "Server error", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3003, debug=True)
