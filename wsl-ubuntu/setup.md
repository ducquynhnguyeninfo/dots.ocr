

## 1. init environment
```
conda create -n dots_ocr python=3.12
conda activate dots_ocr
```

## 2. torch
```
cd dots.ocr
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

```

## 3. flash-attn
```
pip install psutil
pip install flash-attn --no-build-isolation
```

## 4. run setup.py
```
pip install -e . --no-build-isolation
```
## 5. download model weights
