# Deepfake-Minimal Installation Guide (Windows + Mac)

This guide is formatted so you can clearly see what to copy.

Rule:
- Copy only from blocks labeled `COPY THIS`.
- Do not copy normal text paragraphs.

---

## Windows Setup (PowerShell)

### Step 1: Open PowerShell

- Press `Win`
- Search `PowerShell`
- Open it

### Step 2: Run full setup once

`COPY THIS (Windows full setup):`

```powershell
cd "C:\Users\$env:USERNAME\Desktop\Deepfake-minimal"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import numpy, cv2, mediapipe, torch; print('SETUP_OK')"
python src/run_inference.py --file-dialog --model "models/model.pt" --model-type mlp --out "results/showcase_demo.mp4"
```

Expected:
- You should see `SETUP_OK`.
- A file picker opens.
- Select a video or image.
- Output is saved to `results/showcase_demo.mp4`.

### If PowerShell blocks activation

`COPY THIS (Windows activation fix):`

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Daily run command (after first setup)

`COPY THIS (Windows daily run):`

```powershell
cd "C:\Users\$env:USERNAME\Desktop\Deepfake-minimal"
.\.venv\Scripts\Activate.ps1
python src/run_inference.py --file-dialog --model "models/model.pt" --model-type mlp --out "results/showcase_demo.mp4"
```

---

## Mac Setup (Terminal)

### Step 1: Open Terminal

- Press `Cmd + Space`
- Search `Terminal`
- Open it

### Step 2: Install Python 3.11 (one-time)

`COPY THIS (Mac Python install):`

```bash
brew install python@3.11
```

### Step 3: Run full setup once

`COPY THIS (Mac full setup):`

```bash
cd ~/Desktop/Deepfake-minimal
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import numpy, cv2, mediapipe, torch; print('SETUP_OK')"
python src/run_inference.py --file-dialog --model "models/model.pt" --model-type mlp --out "results/showcase_demo.mp4"
```

Expected:
- You should see `SETUP_OK`.
- A file picker opens.
- Select a video or image.
- Output is saved to `results/showcase_demo.mp4`.

### Daily run command (after first setup)

`COPY THIS (Mac daily run):`

```bash
cd ~/Desktop/Deepfake-minimal
source .venv/bin/activate
python src/run_inference.py --file-dialog --model "models/model.pt" --model-type mlp --out "results/showcase_demo.mp4"
```

---

## Common Error Fix (Both Windows and Mac)

Use this only if you get NumPy/OpenCV import errors.

`COPY THIS (NumPy/OpenCV fix):`

```bash
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python numpy
python -m pip install --no-cache-dir numpy==1.26.4 opencv-contrib-python==4.11.0.86 mediapipe==0.10.21
python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```
