"""Classify images with the trained CIFAR-10 model.

Command line:   python predict.py bird.jpg car.jpg
From the web:   server.py imports predict_image() from this file.
"""
import os, sys, threading
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

_HERE = os.path.dirname(os.path.abspath(__file__))
_model = None
_lock = threading.Lock()


def get_model():
    """Load the model once, on first use."""
    global _model
    with _lock:
        if _model is None:
            from keras.models import load_model
            _model = load_model(os.path.join(_HERE, "cifar10_cnn.h5"), compile=False)
    return _model


def predict_image(img):
    """img: a PIL image. Returns all 10 classes, most likely first:
    [{"label": "bird", "confidence": 98.3}, ...]"""
    img = img.convert("RGB").resize((32, 32))
    x = np.expand_dims(np.asarray(img, dtype="float32") / 255.0, 0)
    model = get_model()
    with _lock:
        probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    return [{"label": CLASSES[i], "confidence": round(float(probs[i]) * 100, 2)} for i in order]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict.py <image> [<image> ...]")
    for p in sys.argv[1:]:
        print(f"\n{p}")
        for r in predict_image(Image.open(p))[:3]:
            print(f"  {r['label']:<11} {r['confidence']:6.2f}%")
