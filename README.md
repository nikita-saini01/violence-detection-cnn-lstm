# 🎬 Violence Detection in Videos
### CNN + LSTM Deep Learning Pipeline | Binary Video Classification

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Nikita%20Saini-purple)](https://github.com/YOUR_USERNAME)

> Automated violence detection in video footage using a **TimeDistributed CNN → LSTM** architecture trained from scratch.  
> Trained and evaluated on the **Real Life Violence Situations Dataset** (2,000 videos, 2 classes).

---

## 📊 Results

| Metric | Score |
|---|---|
| Training Accuracy | **88%** |
| Architecture | TimeDistributed CNN + LSTM |
| Dataset Size | 2,000 videos |
| Train / Test Split | 80% / 20% (stratified) |

> **Note:** Model trained from scratch on a custom CNN backbone (no pretransfer learning).  
> Planned improvement: MobileNetV2 pretrained backbone for stronger generalisation — see [Future Improvements](#-future-improvements).

<p align="center">
  <img src="results/training_curves.png" width="700" alt="Training Curves"/>
  <br/>
  <img src="results/confusion_matrix.png" width="400" alt="Confusion Matrix"/>
</p>

---

## 🏗️ Architecture

```
Input Video
    │
    ▼
┌─────────────────────────────────────┐
│  Frame Sampling (np.linspace)        │  8 evenly-spaced frames
│  Resize → 64×64 → Normalize /255    │
└─────────────────────────────────────┘
    │
    ▼ (8, 64, 64, 3)
┌─────────────────────────────────────┐
│  TimeDistributed CNN                 │
│   Block 1: Conv2D(32) + BN + Pool   │
│   Block 2: Conv2D(64) + BN + Pool   │
│   Block 3: Conv2D(128) + BN + Pool  │
│   GlobalAveragePooling2D            │
└─────────────────────────────────────┘
    │
    ▼ (8, 128)
┌─────────────────────────────────────┐
│  LSTM(64)  + Dropout(0.5)           │  temporal sequence learning
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Dense(32, relu) + Dropout(0.3)     │
│  Dense(2, softmax)                  │  Violence / NonViolence
└─────────────────────────────────────┘
```

**Key design choices:**
- `TimeDistributed` wrapper applies the same CNN to every frame independently
- `BatchNormalization` after each conv block for stable, faster training
- `GlobalAveragePooling2D` instead of Flatten — fewer parameters, less overfitting
- `EarlyStopping + ReduceLROnPlateau + ModelCheckpoint` callbacks for robust training
- `float16` array storage for memory efficiency on large video datasets
- Stratified train-test split to preserve class balance

---

## 📁 Project Structure

```
violence-detection-cnn-lstm/
├── violence_detection.ipynb     ← main notebook (training + eval)
├── src/
│   ├── model.py                 ← build_model() function
│   ├── preprocess.py            ← extract_frames() + load_dataset()
│   └── predict.py               ← predict_video() + CLI
├── configs/
│   └── config.yaml              ← all hyperparameters (no hardcoding)
├── results/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/violence-detection-cnn-lstm.git
cd violence-detection-cnn-lstm
pip install -r requirements.txt
```

### 2. Set your dataset path
Edit `configs/config.yaml`:
```yaml
data:
  dataset_path: "path/to/Real Life Violence Dataset/"
```

### 3. Train the model
```bash
# Option A — run the notebook
jupyter notebook violence_detection.ipynb

# Option B — CLI
python src/train.py
```

### 4. Predict on a video
```bash
python src/predict.py --video path/to/video.mp4
```

---

## 📦 Dataset

**Real Life Violence Situations Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

| Class | Videos |
|---|---|
| Violence | 1,000 |
| NonViolence | 1,000 |
| **Total** | **2,000** |

> Dataset is **not included** in this repo. Download from Kaggle and update the path in `configs/config.yaml`.

---

## 🔧 Configuration

All hyperparameters live in `configs/config.yaml`:

```yaml
model:
  img_size: 64
  sequence_length: 8
  lstm_units: 64
  dropout_lstm: 0.5
  dropout_dense: 0.3

training:
  epochs: 30
  batch_size: 4
  learning_rate: 0.0003
  test_size: 0.2
  random_state: 42
```

---

## 🚀 Future Improvements

- [ ] Replace custom CNN with pretrained **MobileNetV2** backbone (transfer learning) — expected to significantly improve generalisation
- [ ] Add **Bidirectional LSTM** for richer temporal context
- [ ] Temporal **attention mechanism** to highlight violence-indicative frames
- [ ] **Grad-CAM** visualizations for model interpretability
- [ ] **Gradio demo** deployment on Hugging Face Spaces
- [ ] K-fold cross-validation for more reliable evaluation
- [ ] Real-time webcam inference

---

## 🛠️ Tech Stack

`TensorFlow/Keras` · `OpenCV` · `NumPy` · `Scikit-Learn` · `Matplotlib` · `Seaborn`

---

## 👤 Author

**Nikita Saini** — B.Tech, IIIT Kota  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
