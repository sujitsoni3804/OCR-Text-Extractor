# 📝 OCR Text Extractor

A web-based OCR tool that extracts and highlights text from images using AI-powered recognition. Upload an image, get labeled text with bounding boxes, and download the annotated result.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Text Detection** | Extracts text from images using EasyOCR with GPU acceleration |
| 📦 **Bounding Boxes** | Draws labeled boxes around detected text regions |
| 🎯 **Confidence Scores** | Shows detection confidence for each text region |
| 🖼️ **Visual Output** | Generates annotated images with labeled text |
| 📥 **Download Results** | Download processed images with highlighted text |
| 📱 **Responsive UI** | Clean web interface for easy image uploads |

---

## 🛠️ Tech Stack

- **Backend:** Flask, Python
- **OCR Engine:** EasyOCR
- **Image Processing:** OpenCV, NumPy
- **ML Framework:** PyTorch (GPU Support)
- **Frontend:** HTML, CSS, JavaScript

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional, for faster processing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/sujitsoni3804/OCR-Text-Extractor.git
cd OCR-Text-Extractor
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install flask opencv-python easyocr torch numpy
```

### Step 4: Run the Application

```bash
python app.py
```

The application will be available at:
- **Local:** http://127.0.0.1:5000
- **Network:** http://YOUR_IP:5000

---

## 🎮 Usage

1. **Upload Image:** Navigate to the homepage and upload an image containing text
2. **Processing:** The system extracts text, draws bounding boxes, and assigns unique labels
3. **View Results:** See detected text with confidence scores and labeled regions
4. **Download:** Save the annotated image with highlighted text areas

---

## 📁 Project Structure

```
OCR-Text-Extractor/
├── app.py                 # Flask application entry point
├── requirements.txt       # Python dependencies
├── pdf_to_images.py       # PDF to image conversion utility
├── templates/
│   └── index.html         # Web interface
├── uploads/               # Uploaded images (temporary)
└── output_images/         # Processed output images
```

---

## 🙏 Acknowledgments

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
- [OpenCV](https://opencv.org/) for image processing
- [PyTorch](https://pytorch.org/) for GPU acceleration

---

<p align="center">
  Made with ❤️ for Text Extraction
</p>
