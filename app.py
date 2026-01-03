from flask import Flask, render_template, request, jsonify, send_file
import cv2
import easyocr
import numpy as np
import torch
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output_images'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

reader = None
CONFIDENCE_THRESHOLD = 0.5

REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080
BASE_FONT_SCALE = 1.4
BASE_FONT_THICKNESS = 3
BASE_BOX_THICKNESS = 3

def init_reader():
    global reader
    if reader is None:
        use_gpu = torch.cuda.is_available()
        reader = easyocr.Reader(['en'], gpu=use_gpu)

def calculate_scale_factor(img_shape):
    height, width = img_shape[:2]
    img_diagonal = np.sqrt(width**2 + height**2)
    ref_diagonal = np.sqrt(REFERENCE_WIDTH**2 + REFERENCE_HEIGHT**2)
    
    scale_factor = img_diagonal / ref_diagonal
    scale_factor = max(0.3, min(scale_factor, 3.0))
    
    return scale_factor

def get_dynamic_params(img_shape):
    """Get dynamically scaled parameters based on image size"""
    scale = calculate_scale_factor(img_shape)
    
    return {
        'font_scale': BASE_FONT_SCALE * scale,
        'font_thickness': max(1, int(BASE_FONT_THICKNESS * scale)),
        'box_thickness': max(2, int(BASE_BOX_THICKNESS * scale)),
        'margin': max(5, int(8 * scale))
    }

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(denoised)

def generate_unique_id(index):
    result, index = "", index + 1
    while index > 0:
        index -= 1
        result = chr(97 + (index % 26)) + result
        index //= 26
    return result

def check_overlap(new_rect, existing_rects, margin=3):
    x1, y1, x2, y2 = [new_rect[i] + [-margin, -margin, margin, margin][i] for i in range(4)]
    for ex_x1, ex_y1, ex_x2, ex_y2 in existing_rects:
        if not (x2 < ex_x1 or x1 > ex_x2 or y2 < ex_y1 or y1 > ex_y2):
            return True
    return False

def find_label_position(bbox_coords, text_size, existing_labels, img_shape, margin):
    height, width = img_shape[:2]
    text_width, text_height = text_size
    x_min, y_min, x_max, y_max = bbox_coords
    
    positions = [
        (x_min, y_min - margin, 'top'),
        (x_min, y_max + text_height + margin, 'bottom'),
        (x_max + margin, y_min + text_height, 'right'),
        (x_min - text_width - margin, y_min + text_height, 'left'),
    ]
    
    for pos_x, pos_y, _ in positions:
        label_rect = (pos_x, pos_y - text_height, pos_x + text_width, pos_y)
        
        if 0 <= label_rect[0] and 0 <= label_rect[1] and label_rect[2] <= width and label_rect[3] <= height:
            bbox_rect = (x_min, y_min, x_max, y_max)
            if not check_overlap(label_rect, [bbox_rect]) and not check_overlap(label_rect, existing_labels):
                return pos_x, pos_y, label_rect
    
    default_y = y_min - margin if y_min > text_height + margin else y_max + text_height + margin
    default_rect = (x_min, default_y - text_height, x_min + text_width, default_y)
    return x_min, default_y, default_rect

def draw_label(img, text, bbox_coords, existing_labels, params):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = params['font_scale']
    thickness = params['font_thickness']
    margin = params['margin']
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x_min, y_min, x_max, y_max = bbox_coords
    
    final_x, final_y, label_rect = find_label_position(bbox_coords, text_size, existing_labels, img.shape, margin)
    
    cv2.putText(img, text, (int(final_x), int(final_y)), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    return label_rect

def process_image(image_path):
    init_reader()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    original = img.copy()
    params = get_dynamic_params(original.shape)
    
    upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    processed = preprocess_image(upscaled)
    
    detections = reader.readtext(processed, detail=1, paragraph=False, min_size=5,
                                text_threshold=0.5, low_text=0.3, link_threshold=0.3,
                                canvas_size=2800, mag_ratio=1.5)
    
    results, existing_labels = [], []
    
    for idx, (bbox, text, conf) in enumerate(detections):
        text = text.strip()
        if not text:
            continue

        conf_val = float(conf)
        if conf_val > 1.0:
            conf_val = conf_val / 100.0

        if conf_val < CONFIDENCE_THRESHOLD:
            continue

        points = (np.array(bbox, dtype=np.int32) // 2)
        x, y = int(np.min(points[:, 0])), int(np.min(points[:, 1]))
        w, h = int(np.max(points[:, 0]) - x), int(np.max(points[:, 1]) - y)
        unique_id = generate_unique_id(idx)
        
        cv2.polylines(original, [points], True, (0, 255, 0), params['box_thickness'])
        label_rect = draw_label(original, f"{unique_id}: {text}", (x, y, x + w, y + h),
                               existing_labels, params)
        existing_labels.append(label_rect)
        
        results.append({
            'id': unique_id,
            'text': text,
            'confidence': round(conf_val, 4),
            'bbox': {'x': x, 'y': y, 'width': w, 'height': h}
        })
    
    return original, results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = f"output_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    file.save(input_path)
    
    try:
        processed_img, results = process_image(input_path)
        cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_base64}",
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)