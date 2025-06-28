import fitz  # PyMuPDF
from PIL import Image
from main import pytesseract 
import io
import cv2
import numpy as np
import re


# --- REFINEMENT TOOLS ---
def detect_math_symbols(text):
    math_regex = r'[=+\-*/^(){}\[\]∫Σ√∞πθ±×÷≠≤≥∑∏∂√∞≈∝]'
    return bool(re.search(math_regex, text))


def refine_text(text):
    # Remove ASCII garbage, excessive special characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    text = re.sub(r'[^\w\s.,;:()\[\]{}\-+=*/^=]', '', text)  # retain math friendly chars
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    return text.strip()


def analyze_text(text):
    summary = []
    if detect_math_symbols(text):
        summary.append("[MATH SYMBOLS DETECTED]")
    return "\n".join(summary) if summary else "[No special patterns found]"



def detect_contours(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_count = len(contours)
    if contour_count > 20:  # Heuristic: many contours = likely diagram
        return f"[DIAGRAM or SHAPE CONTENT DETECTED] ({contour_count} contours)"
    else:
        return "[No major shapes/diagrams detected]"


def detect_images_in_page(page):
    images = page.get_images(full=True)
    image_count = len(images)
    if image_count > 0:
        return f"[DIAGRAM or IMAGE DETECTED] ({image_count} images on this page)"
    return "[No diagrams/images found]"


def handle_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        full_text = ""

        for i, page in enumerate(doc):
            print(f"🔍 Processing Page {i+1}")
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)

            refined = refine_text(text)
            analysis = analyze_text(refined)
            image_check = detect_images_in_page(page)

            full_text += f"\n\n--- Page {i+1} ---\n{image_check}\n{analysis}\n{refined}"

        return full_text.strip()

    except Exception as e:
        return f"❌ Error processing PDF: {e}"



def handle_image(file_path):
    try:
        print("📸 OCR running on:", file_path)
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        refined = refine_text(text)
        analysis = analyze_text(refined)
        shape_check = detect_contours(file_path)

        return f"{shape_check}\n{analysis}\n\n{refined}"

    except Exception as e:
        return f"❌ Error processing image: {str(e)}"