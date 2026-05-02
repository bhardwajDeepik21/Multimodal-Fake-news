A hybrid deep learning system that fuses text (RoBERTa) and image (ViT-B/16) modalities using multi-head cross-attention for fake news detection on the Fakeddit dataset.
key results


Project Structure
fake_news_detection/
├── configs/
│   └── config.yaml              # All hyperparameters
├── data/
│   ├── dataset.py               # FakedditDataset class
│   └── preprocessing.py         # Text/image preprocessing
├── models/
│   ├── text_encoder.py          # RoBERTa encoder
│   ├── vision_encoder.py        # ViT-B/16 encoder
│   ├── cross_attention.py       # Multi-head cross-attention fusion
│   ├── adversarial.py           # Domain discriminator
│   ├── classifier.py            # MLP classifier head
│   └── multimodal_model.py      # Full pipeline
├── utils/
│   ├── metrics.py               # Accuracy, F1, AUC
│   ├── losses.py                # BCE + Focal Loss
│   └── augmentation.py          # Back-translation, image jitter
├── explainability/
│   ├── gradcam.py               # Grad-CAM on ViT
│   └── attention_viz.py         # Cross-attention weight visualization
├── scripts/
│   ├── download_fakeddit.py     # Dataset download helper
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation + ablation
│   └── inference.py             # Single-sample inference
├── notebooks/
│   └── exploration.ipynb        # EDA notebook
├── requirements.txt
└── README.md

Setup
bash# 1. Clone and enter project
git clone <repo-url>
cd fake_news_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download Fakeddit dataset
python scripts/download_fakeddit.py

# 4. Train the model
python scripts/train.py --config configs/config.yaml

# 5. Evaluate + run ablation
python scripts/evaluate.py --checkpoint results/best_model.pt

# 6. Single sample inference
python scripts/inference.py --text "Breaking: ..." --image path/to/image.jpg

Architecture
Text → [RoBERTa-base] → 768-dim CLS embedding
                                       ↓
                          [Multi-Head Cross-Attention]  ← 768-dim patch embeddings ← [ViT-B/16] ← Image
                                       ↓
                          [Gated Fusion + MLP Head]
                                       ↓
                              [Binary / 6-class Output]

Ablation Studies
Run all ablation variants:
bashpython scripts/evaluate.py --ablation all
Variants tested:

Text-only (RoBERTa)
Image-only (ViT)
Late-fusion (concat)
Cross-attention (ours)
Cross-attention + adversarial (full model)


Explainability
bashpython scripts/inference.py --text "..." --image img.jpg --explain
Generates:

Grad-CAM heatmaps on image regions
Cross-attention weight matrices
SHAP values for text tokens


Reproducibility

Random seed fixed at 42 everywhere
All experiments tracked with Weights & Biases
requirements.txt pins all versions
Trained model weights released on HuggingFace Hub

