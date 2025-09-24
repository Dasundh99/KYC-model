# app.py
import io
import traceback
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)
CORS(app)  # allow requests from the frontend

# --- Config / thresholds ---
# Cosine similarity threshold: 1.0 is identical, -1 is opposite.
# Tune for your data. 0.7-0.9 are common values depending on model and quality.
SIMILARITY_THRESHOLD = 0.75
ID_TO_ID_THRESHOLD = 0.7   # how strictly the two ID sides must match each other (optional)

# --- Initialize models (runs on CPU unless you have CUDA and torch configured) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)  # face detector / cropper
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # embedding model


def read_image_from_file_storage(file_storage):
    """Read Flask FileStorage to PIL.Image"""
    try:
        file_bytes = file_storage.read()
        return Image.open(io.BytesIO(file_bytes)).convert('RGB')
    except Exception:
        return None


def get_embedding(img_pil):
    """
    Given a PIL image, detect the *largest/primary* face with MTCNN,
    crop/align it and return a normalized embedding tensor (1D).
    Returns None if no face found.
    """
    # mtcnn returns a torch tensor (C,H,W) or None
    try:
        face_tensor = mtcnn(img_pil)  # returns a cropped tensor or None
        if face_tensor is None:
            return None
        # Ensure batch dim
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = resnet(face_tensor)  # shape (1, 512)
            emb = F.normalize(emb, p=2, dim=1)  # L2 normalize
            return emb.squeeze(0)  # 1D tensor
    except Exception:
        traceback.print_exc()
        return None


def cosine_similarity(a, b):
    """Compute cosine similarity between two 1D normalized torch tensors"""
    if a is None or b is None:
        return None
    # ensure they're 1D and float tensors on same device
    a = a.to(device)
    b = b.to(device)
    sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return sim


@app.route('/verify', methods=['POST'])
def verify():
    """
    Accepts multipart/form-data with fields:
      - id_front: file
      - id_back: file
      - selfie: file

    Returns JSON:
      { verified: bool, reason: str, similarity: {...} }
    """
    try:
        # Get files
        id_front_f = request.files.get('id_front')
        id_back_f = request.files.get('id_back')
        selfie_f = request.files.get('selfie')

        if not id_front_f or not id_back_f or not selfie_f:
            return jsonify({'verified': False, 'reason': 'Missing one or more files (id_front, id_back, selfie)'}), 400

        # Read images
        id_front_img = read_image_from_file_storage(id_front_f)
        id_back_img = read_image_from_file_storage(id_back_f)
        selfie_img = read_image_from_file_storage(selfie_f)

        if id_front_img is None or id_back_img is None or selfie_img is None:
            return jsonify({'verified': False, 'reason': 'Failed to read one or more images (ensure valid image files)'}), 400

        # Get embeddings
        emb_front = get_embedding(id_front_img)
        emb_back = get_embedding(id_back_img)
        emb_selfie = get_embedding(selfie_img)

        if emb_front is None:
            return jsonify({'verified': False, 'reason': 'No face detected in id_front'}), 400
        if emb_back is None:
            return jsonify({'verified': False, 'reason': 'No face detected in id_back'}), 400
        if emb_selfie is None:
            return jsonify({'verified': False, 'reason': 'No face detected in selfie'}), 400

        # Compute similarities
        sim_front_back = cosine_similarity(emb_front, emb_back)
        sim_selfie_front = cosine_similarity(emb_selfie, emb_front)
        sim_selfie_back = cosine_similarity(emb_selfie, emb_back)

        # Decide verification:
        # Approach:
        #  1) Optionally require the two ID sides to be consistent (front/back)
        #  2) Require selfie to match either front or back above threshold
        id_consistent = (sim_front_back is not None and sim_front_back >= ID_TO_ID_THRESHOLD)
        selfie_matches_front = sim_selfie_front is not None and sim_selfie_front >= SIMILARITY_THRESHOLD
        selfie_matches_back = sim_selfie_back is not None and sim_selfie_back >= SIMILARITY_THRESHOLD

        verified = False
        reason = ''
        if not id_consistent:
            # If ID sides are not consistent we might still proceed, but we warn user.
            # Here we choose to still allow verification if selfie strongly matches at least one side.
            if selfie_matches_front or selfie_matches_back:
                verified = True
                reason = 'IDs not consistent, but selfie matches one side.'
            else:
                verified = False
                reason = 'ID front and back do not match each other, and selfie does not match either.'
        else:
            # IDs consistent; require selfie match
            if selfie_matches_front or selfie_matches_back:
                verified = True
                reason = 'Verified: selfie matches ID.'
            else:
                verified = False
                reason = 'Selfie does not match ID.'

        response = {
            'verified': verified,
            'reason': reason,
            'similarity': {
                'id_front_vs_id_back': sim_front_back,
                'selfie_vs_id_front': sim_selfie_front,
                'selfie_vs_id_back': sim_selfie_back
            },
            'thresholds': {
                'similarity_threshold': SIMILARITY_THRESHOLD,
                'id_to_id_threshold': ID_TO_ID_THRESHOLD
            }
        }
        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'verified': False, 'reason': 'Server error', 'error': str(e)}), 500


if __name__ == '__main__':
    # Run always on port 5000
    # In production you should use gunicorn or a process manager — this keeps it simple.
    app.run(host='0.0.0.0', port=5000, debug=False)
