import pytesseract
import os

# Try common macOS installation paths for Tesseract
possible_paths = [
    "/usr/local/bin/tesseract",   # Intel Macs (default Homebrew path)
    "/opt/homebrew/bin/tesseract" # Apple Silicon (M1/M2/M3)
]

# Pick the first path that exists
tesseract_path = None
for path in possible_paths:
    if os.path.exists(path):
        tesseract_path = path
        break

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print("✅ Tesseract path set:", tesseract_path)
else:
    print("❌ Tesseract not found. Try running: brew install tesseract")
