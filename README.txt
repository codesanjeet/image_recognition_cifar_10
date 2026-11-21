RUN THE CIFAR-10 IMAGE RECOGNITION PROJECT ON LINUX
====================================================

1. Install Python (skip if `python3 --version` already works)
     Ubuntu/Debian/Mint:  sudo apt update && sudo apt install -y python3 python3-venv python3-pip
     Fedora:              sudo dnf install -y python3 python3-pip
     Arch:                sudo pacman -S python python-pip

2. Open a terminal in this folder and run the setup ONCE (about 5 min, ~1 GB download):
     bash setup_linux.sh

3. Every time you want to use it:
     source cifar_env/bin/activate
     python predict.py bird.jpg
     python predict.py car.jpg my_photo.png        # any images you like

4. (Optional) Use the Jupyter notebook:
     source cifar_env/bin/activate
     jupyter notebook
   Then open cifar10_image_recognition.ipynb.
   - Cell 1 (!pip install ...) can be skipped, setup already did it.
   - Training cell needs the CIFAR-10 files in ~/.keras/datasets/cifar-10-batches-py
     (data_batch_1..5 and test_batch). You do NOT need to retrain: cifar10_cnn.h5 is
     already trained.
   - The last cell predicts 'frog.jpg', which isn't in your zip. Change it to bird.jpg
     or any image you have.

The model only knows 10 classes: airplane, automobile, bird, cat, deer, dog, frog,
horse, ship, truck. Images are shrunk to 32x32, so it works best on a clear photo of
one object.
