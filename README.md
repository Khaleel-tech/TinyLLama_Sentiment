# 🧠 Sentiment Analysis with TinyLlama 1.1B — QLoRA Fine-tuning
 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/tinyllama-sentiment-qlora/blob/main/tinyllama_sentiment.ipynb)
 
## 📌 Overview
 
Fine-tuned **TinyLlama 1.1B** using **QLoRA** on the **SST-2** dataset for binary sentiment classification (positive/negative). Achieved a **+44% accuracy improvement** over the base model by training only **0.4% of the model's parameters** in under 8 minutes on a free T4 GPU.
 
---
 
## 📊 Results
 
| | Accuracy |
|---|---|
| 🔴 Baseline (no fine-tuning) | 52.00% |
| 🟢 After QLoRA fine-tuning | **96.00%** |
| 📈 Improvement | **+44%** |
 
---
 
## 🗂️ Project Structure
 
```
tinyllama-sentiment-qlora/
├── tinyllama_sentiment.ipynb   # Full Colab notebook
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```
 
---
 
## 🛠️ Tech Stack
 
| Tool | Purpose |
|---|---|
| TinyLlama 1.1B | Base language model |
| QLoRA (4-bit) | Memory-efficient fine-tuning |
| PEFT | LoRA adapter management |
| TRL / SFTTrainer | Supervised fine-tuning |
| HuggingFace Datasets | SST-2 dataset loading |
| Google Colab T4 GPU | Free training environment |
 
---
 
## 🚀 How It Works
 
### 1. Dataset
- **SST-2** (Stanford Sentiment Treebank) — movie reviews labeled positive/negative
- Used **1,000 training samples** and **200 validation samples**
### 2. Prompt Format
```
### Instruction:
Classify the sentiment of the following movie review as positive or negative.
 
### Review:
{movie review text}
 
### Sentiment:
{positive/negative}
```
 
### 3. QLoRA Fine-tuning
- Froze the base TinyLlama 1.1B model
- Added LoRA adapters to attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- Trained only **4.5M out of 1.1B parameters (0.4%)**
- 5 epochs, ~8 minutes on T4 GPU
### 4. Evaluation
- Fed each review to the model and extracted "positive" or "negative" from output
- Compared predicted vs actual label across 200 validation samples
---
 
## ⚙️ LoRA Configuration
 
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```
 
---
 
## 📦 Installation
 
```bash
pip install transformers==4.45.0 peft==0.13.0 trl==0.8.6 bitsandbytes==0.49.2 accelerate datasets evaluate
```
 
---
 
## 🔁 Run It Yourself
 
1. Open the notebook in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4)
3. Run all cells in order
4. See baseline → fine-tuned comparison at the end
---
 
## 📈 Training Loss
 
| Step | Loss |
|---|---|
| 20 | 2.229 |
| 60 | 0.932 |
| 160 | 0.802 |
| 300 | 0.758 |
 
---
 
## 👤 Author
 
**Khasim Khaleel Basha**
B.Tech AI & Data Science — KL University
[LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)
