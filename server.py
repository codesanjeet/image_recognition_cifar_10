"""Web server: serves the page and sends uploaded images to predict.py.

Start:  uvicorn server:app --port 8000
Open:   http://127.0.0.1:8000
"""
import io, os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from predict import get_model, predict_image

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="CIFAR-10 image recognition")
# Lets the page work even if you open the .html file directly from disk.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def warm_up():
    get_model()  # load the model now so the first upload is fast


@app.get("/")
def home():
    return FileResponse(os.path.join(HERE, "image_recognition.html"))


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 15 MB. Choose a smaller image.")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # fixes rotated phone photos
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file is not an image this app can read. Use a JPG, PNG or WebP.")
    return {"filename": file.filename, "results": predict_image(img)}
