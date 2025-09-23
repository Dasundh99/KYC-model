import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins

# Configure upload folder and max upload size (16 MB)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

@app.route("/verify", methods=["POST"])
def verify_wallet():
    try:
        # Receive files
        id_front = request.files.get("id_front")
        id_back = request.files.get("id_back")  # optional now
        selfie = request.files.get("selfie")

        print("Received ID Front:", id_front)
        print("Received Selfie:", selfie)

        if not id_front or not selfie:
            return jsonify({"message": "Missing required files", "status": "error"}), 400

        # Save files temporarily
        id_front_path = os.path.join(UPLOAD_FOLDER, f"id_front_{id_front.filename}")
        selfie_path = os.path.join(UPLOAD_FOLDER, f"selfie_{selfie.filename}")

        id_front.save(id_front_path)
        selfie.save(selfie_path)

        # Load images
        id_front_image = face_recognition.load_image_file(id_front_path)
        selfie_image = face_recognition.load_image_file(selfie_path)

        # Get face encodings
        id_front_enc = face_recognition.face_encodings(id_front_image)
        selfie_enc = face_recognition.face_encodings(selfie_image)

        if len(id_front_enc) == 0 or len(selfie_enc) == 0:
            return jsonify({
                "message": "Face not detected in ID front or selfie",
                "status": "error"
            }), 400

        # Compare selfie against ID front
        match_front = bool(face_recognition.compare_faces([id_front_enc[0]], selfie_enc[0], tolerance=0.6)[0])

        return jsonify({
            "message": "Verification complete",
            "status": "success",
            "verified": match_front,
            "match_front": match_front
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"message": "Server error", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
