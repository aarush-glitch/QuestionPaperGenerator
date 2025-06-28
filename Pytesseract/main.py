import pytesseract
import os

# Manually set the full path again
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Optional debug print to confirm
if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
    print("❌ Tesseract not found at that path!")
else:
    print("✅ Tesseract path set:", pytesseract.pytesseract.tesseract_cmd)
