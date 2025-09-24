# file: id_ocr.py
import io
import re
import base64
from typing import Tuple, List, Dict, Any
from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image
import easyocr

app = Flask(__name__)
reader = easyocr.Reader(['en'], gpu=False)

# ---------- Preprocessing helpers ----------
def read_image_from_filestorage(fs) -> np.ndarray:
    data = fs.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def resize_for_processing(img: np.ndarray, max_dim=1200) -> np.ndarray:
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_document_contour(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return orig, None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            try:
                warped = four_point_transform(orig, approx.reshape(4,2))
                return warped, approx.reshape(4,2)
            except Exception:
                continue
    return orig, None

def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 15, 10)
    th = cv2.medianBlur(th, 3)
    return th

def image_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')

# ---------- OCR ----------
def ocr_image(img: np.ndarray) -> List[Dict[str, Any]]:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim==3 else cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb, detail=1)
    structured = []
    for bbox, text, conf in results:
        structured.append({
            'text': text,
            'confidence': float(conf),
            'bbox': [ [int(pt[0]), int(pt[1])] for pt in bbox ]
        })
    return structured

# ---------- Field extraction ----------
RE_ID_NUMBER = re.compile(r'([A-Z0-9\-]{4,20})')
RE_DATE = re.compile(r'((?:\d{2}[\/\-\.\s]\d{2}[\/\-\.\s]\d{2,4})|(?:\d{4}[\/\-\.\s]\d{2}[\/\-\.\s]\d{2}))')

def extract_fields_from_text(raw_text: str, lines: List[Dict]) -> Dict[str, str]:
    text = raw_text.replace('\n',' ').strip()
    fields = {'name': None, 'id_number': None, 'dob': None}
    date_match = RE_DATE.search(text)
    if date_match:
        fields['dob'] = date_match.group(0)
    for l in lines:
        t = l['text']
        tl = t.lower()
        if 'id' in tl and len(t.split()) <= 4:
            m = RE_ID_NUMBER.search(t)
            if m:
                fields['id_number'] = m.group(1)
                break
    if not fields['id_number']:
        m = RE_ID_NUMBER.search(text)
        if m:
            fields['id_number'] = m.group(1)
    candidate = None
    for l in lines:
        t = l['text'].strip()
        if len(t.split()) >= 2 and not any(char.isdigit() for char in t):
            if not candidate or len(t) > len(candidate):
                candidate = t
    if candidate:
        fields['name'] = candidate
    return fields

# ---------- Flask endpoint ----------
@app.route('/read_id', methods=['POST'])
def read_id():
    """
    Accepts form-data with fields 'id_front', 'id_back', 'selfie'
    Returns JSON with OCR results for each image.
    """
    images = {}
    for key in ['id_front', 'id_back', 'selfie']:
        if key not in request.files:
            return jsonify({'error': f'no file uploaded for {key}'}), 400
        try:
            images[key] = read_image_from_filestorage(request.files[key])
        except Exception as e:
            return jsonify({'error': f'cannot read {key}', 'detail': str(e)}), 400

    results = {}
    for key, img in images.items():
        img_resized = resize_for_processing(img, max_dim=1200)
        warped, contour = detect_document_contour(img_resized)
        preproc = preprocess_for_ocr(warped)
        ocr_on_warped = ocr_image(warped)
        preproc_bgr = cv2.cvtColor(preproc, cv2.COLOR_GRAY2BGR)
        ocr_on_preproc = ocr_image(preproc_bgr)

        combined_lines = ocr_on_warped + ocr_on_preproc
        seen = set()
        merged = []
        for item in sorted(combined_lines, key=lambda x: -x['confidence']):
            txt = item['text'].strip()
            if not txt or txt.lower() in seen:
                continue
            seen.add(txt.lower())
            merged.append(item)

        raw_text = "\n".join([it['text'] for it in merged])
        fields = extract_fields_from_text(raw_text, merged)

        results[key] = {
            'raw_text': raw_text,
            'lines': merged,
            'fields': fields,
            'has_document_contour': contour is not None,
            'debug_image_cropped': image_to_base64(warped),
            'debug_image_preprocessed': image_to_base64(preproc_bgr)
        }

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5003, debug=True)
