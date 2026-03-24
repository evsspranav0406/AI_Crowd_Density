# AI Crowd Density - Fix libGL.so.1 OpenCV Import Error on Streamlit Cloud

## Approved Plan Steps (Deployment: Streamlit Cloud)

### 1. [x] Update requirements.txt
- Replace `opencv-python` with `opencv-python-headless`
- Add explicit CPU PyTorch: `torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- Test local install and run `streamlit run app.py`

### 2. [x] Create packages.txt for Streamlit Cloud system dependencies
- Add: `libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1`
- (Ensures libGL.so.1 available)

### 3. [x] Test locally
- `pip install -r requirements.txt`
- `streamlit run app.py` (verify no import error, webcam/video works)

### 4. [ ] Commit & Deploy
- Push to GitHub
- Redeploy on Streamlit Cloud (auto-detects packages.txt/requirements.txt)

### 5. [ ] Verify Deployment
- Check Streamlit Cloud app health
- Test upload video/webcam for density metrics/heatmap

**Next: Step 4 - Commit & Deploy**
