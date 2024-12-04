# 🌟 ABSA Compound Toolkit 🌟

Welcome to the **ABSA Compound Toolkit**—your comprehensive solution for training and evaluating compound **Aspect-Based Sentiment Analysis (ABSA)** tasks! 🚀 This toolkit integrates cutting-edge **ASQP** (Aspect Sentiment Quadruple Prediction) approaches and provides detailed evaluation tools to empower your research and development. 💡✨

---

## 📖 What is ABSA Compound Toolkit?

The ABSA Compound Toolkit simplifies the process of training and evaluating ABSA models by integrating state-of-the-art approaches. Key features include:

- 🛠️ **Training Frameworks** for the following ASQP methods:
  - **MVP** 🥇  
    Based on the ACL 2023 paper [MvP: Multi-view Prompting Improves Aspect Sentiment Tuple Prediction](https://arxiv.org/abs/2305.12627).  
    GitHub repository: [multi-view-prompting](https://github.com/ZubinGou/multi-view-prompting)  
  - **Paraphrase** 🔄  
    As described in the paper [A Paraphrase Generation-Based Approach for Aspect Sentiment Quad Prediction](https://arxiv.org/pdf/2110.00796v1.pdf).  
    GitHub repository: [absa-quad](https://github.com/isakzhang/absa-quad)  
  - **DLO** 🌐  
    Based on the paper [DLO: A Scalable Solution for ABSA Quadruple Prediction](https://arxiv.org/pdf/2210.10291v1.pdf).  
    GitHub repository: [AspectQuad](https://github.com/hmt2014/AspectQuad/tree/main)  

- 📊 **Post-Training Analytics**:  
  - Generates a detailed `JSON` report for every training session containing:
    - **Evaluation Metrics**: F1, Precision, Recall, and more! 📈  
    - **Gold Labels**: The ground truth data ✅  
    - **Predicted Labels**: The model's predictions 🤖  

---

## 🚀 Features at a Glance

✔️ Supports multiple state-of-the-art **ASQP** approaches.  
✔️ Automatically generates **comprehensive evaluation reports** post-training.  
✔️ Flexible and extensible design for both researchers and developers.  
✔️ Seamless integration with modern **NLP libraries** for efficient processing.  

---

## 🏗️ Installation Guide

### Requirements
Make sure you have the necessary dependencies installed. 🛠️ You can find them listed in the requirement files.

### Step-by-Step Installation

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/NilsHellwig/ABSA-compound-toolkit.git
   cd ABSA-compound-toolkit