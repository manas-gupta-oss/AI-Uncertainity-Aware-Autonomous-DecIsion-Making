# 🚀 AI Uncertainty-Aware Autonomous Decision Making

An advanced deep learning project focused on **uncertainty-aware decision making** using **Monte Carlo Dropout (MC Dropout)** to improve reliability and safety in AI systems.

---

## 📌 Overview

Traditional AI models make predictions **without knowing when they are wrong**.  
This project addresses that limitation by enabling models to:

- Quantify **uncertainty**
- Detect **unsafe decisions**
- Make **risk-aware predictions**

This is especially important for real-world applications where wrong decisions can be costly or dangerous.

---

## 🧠 Key Idea

Instead of making a single prediction, the model performs **multiple stochastic forward passes** using MC Dropout to estimate uncertainty.

From these predictions, we compute:

- **Mean Prediction**
- **Entropy (H)** → Total uncertainty  
- **Mutual Information (MI)** → Epistemic uncertainty  

---

## ⚙️ Features

- Uncertainty estimation using MC Dropout  
- Risk-aware decision mechanism  
- Detection of unsafe predictions  
- Mutual Information-based uncertainty scoring  
- Modular and clean PyTorch implementation  

---


---

## 🔬 Methodology

### 1. Monte Carlo Dropout
- Enable dropout during inference  
- Perform multiple forward passes  
- Approximate Bayesian inference  

### 2. Uncertainty Metrics
- **Entropy (H):** Measures total uncertainty  
- **Mutual Information (MI):** Captures model (epistemic) uncertainty  

### 3. Risk-Aware Decision Rule


def risk_aware_decision(pred_class, mi, tau=0.5):
    if mi > tau:
        return "CONSERVATIVE_SIGNAL"
    return f"ACTION_{pred_class}"


