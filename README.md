# Smart Product Intelligence Capstone

An end-to-end deep learning project spanning structured data, computer vision, natural language processing, large language models, and generative models, built on the **Amazon Reviews 2023 (McAuley Lab)** dataset using the `All_Beauty` category.

---

## 🛠️ Technology Stack
* **Framework**: TensorFlow & Keras
* **Datasets & Transformers**: Hugging Face `datasets` & `transformers`
* **Generative Models**: Hugging Face `diffusers` (Stable Diffusion)
* **Interactive Demo**: Gradio / Streamlit
* **Environment Manager**: Conda / Python 3.10

---

## 📂 Repository Structure
```
smart-product-intelligence/
├── README.md               # Project documentation
├── requirements.txt         # Dependency declarations
├── data/                    # Cached dataset splits and images (gitignored)
├── notebooks/               # Step-by-step milestone execution notebooks
│   ├── 00_eda.ipynb                     # Milestone 0: EDA & Prep
│   ├── 01_tabular_mlp.ipynb             # Milestone 1: Tabular MLP
│   ├── 02_vision_cnn_transfer.ipynb     # Milestone 2: Computer Vision
│   ├── 03_text_embeddings.ipynb         # Milestone 3: Text Vectorization
│   ├── 04_transformers.ipynb            # Milestone 4: Transformers
│   ├── 05_llm_rag_finetune.ipynb        # Milestone 5: LLMs & Fine-tuning
│   └── 06_diffusion.ipynb               # Milestone 6: Stable Diffusion
├── src/                     # Reusable production-grade scripts
│   ├── data.py                          # Data utilities
│   ├── models.py                        # Model definitions
│   └── utils.py                         # Helper functions
└── app/                     # Web deployment
    └── app.py                           # Gradio/Streamlit UI application
```

---

## 🚀 Milestone Roadmap
* **Milestone 0**: Data Preparation, cleaning, product-level splitting, and EDA.
* **Milestone 1**: Tabular Multi-Layer Perceptron (MLP) for rating prediction.
* **Milestone 2**: Category prediction from product images using CNN (scratch vs. Transfer Learning).
* **Milestone 3**: Review sentiment classification (Bag-of-Words vs. Embeddings) & Semantic Product Search.
* **Milestone 4**: Transformer-based text classifier (fine-tuning BERT/DistilBERT).
* **Milestone 5**: Grounded LLM Assistant (RAG) & structured review summarization.
* **Milestone 6**: Generative AI (Stable Diffusion) for alternate product lifestyle image generation.
* **Milestone 7**: Integration of all components into a single **Smart Product Assistant** Gradio/Streamlit app.

---

## ⚙️ Setup and Installation

### 1. Clone & Environment setup
```bash
# Create conda environment
conda create --prefix ./.venv python=3.10 -y
conda activate ./.venv

# Install dependencies
pip install -r requirements.txt
```
