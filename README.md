# 🎓 NLP Academy - Interactive Educational Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/spaces/aradhyapavan/nlp-ultimate-tutor)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

> **A comprehensive Natural Language Processing educational platform that bridges the gap between theoretical concepts and practical implementation.**

## 🌟 Live Demo

🚀 **[Try NLP Academy on Hugging Face Spaces](https://huggingface.co/spaces/aradhyapavan/nlp-ultimate-tutor)**

## 📖 About

NLP Academy is an interactive educational platform designed to make Natural Language Processing concepts accessible to everyone. Built by an AI/ML graduate with NLP specialization from BITS Pilani, India, this platform provides hands-on demonstrations of core NLP techniques, from basic text processing to sophisticated transformer applications.

### 🎯 Mission
To democratize NLP education by providing interactive, visual, and practical learning experiences that help students, researchers, and developers truly understand how modern language processing works.

## 📁 Project Structure

```
NLP_Academy/
├── 📱 app.py                     # Main Flask application
├── 🐳 Dockerfile               # Docker configuration
├── 📋 requirements.txt         # Python dependencies
├── 📝 README.md               # HF Spaces documentation
├── 📂 components/             # NLP module implementations
│   ├── 🔤 preprocessing.py    # Text cleaning & normalization
│   ├── 🎯 tokenization.py     # Token extraction & analysis
│   ├── 🏷️ pos_tagging.py      # Part-of-speech tagging
│   ├── 👤 named_entity.py     # Named entity recognition
│   ├── 💭 sentiment.py        # Sentiment & emotion analysis
│   ├── 📝 summarization.py    # Text summarization (extractive/abstractive)
│   ├── 📊 topic_analysis.py   # Topic modeling & analysis
│   ├── ❓ question_answering.py # QA systems
│   ├── ✍️ text_generation.py  # Text generation with GPT-2
│   ├── 🌍 translation.py      # Multi-language translation
│   ├── 📂 classification.py   # Text classification & zero-shot
│   └── 🔍 vector_embeddings.py # Semantic search & embeddings
├── 📂 utils/                  # Utility functions
│   ├── 🔧 model_loader_hf.py  # Optimized model loading
│   ├── 🛠️ helpers.py          # Helper functions
│   └── 📊 visualization.py    # Plotting utilities
├── 📂 templates/             # HTML templates
│   ├── 🏠 index.html         # Main interface
│   ├── 🎨 base.html          # Base template
│   └── 📄 [module].html      # Individual module pages
└── 📂 static/               # Static assets
    ├── 🎨 css/              # Stylesheets
    └── ⚡ js/               # JavaScript files
```

## ✨ Current Features - 12 NLP Modules

| Module | Features | Models/Algorithms | Tools & Libraries |
|--------|----------|-------------------|-------------------|
| **🔤 Preprocessing** | Text cleaning, normalization, stopword removal, stemming, lemmatization, n-grams, word clouds | Rule-based algorithms, Porter Stemmer, WordNet Lemmatizer | NLTK, re, string, WordCloud, Counter |
| **🎯 Tokenization** | Word/sentence tokenization, token analysis, linguistic features, WordPiece subword tokenization | NLTK tokenizers, spaCy tokenizers, WordPiece | NLTK, spaCy, transformers, collections |
| **🏷️ POS Tagging** | Part-of-speech tagging, grammatical analysis, dependency parsing, visualization | NLTK Perceptron Tagger, spaCy en_core_web_sm | spaCy, NLTK, matplotlib, pandas |
| **👤 Named Entity** | Person, organization, location extraction, entity relationships, confidence scoring | dslim/bert-base-NER, spaCy en_core_web_sm NER | transformers, spaCy, NetworkX, matplotlib |
| **💭 Sentiment Analysis** | Sentiment polarity, emotion detection, sentence-level analysis, confidence metrics | NLTK VADER, DistilBERT-SST2, cardiffnlp/twitter-roberta-base-emotion | NLTK, transformers, matplotlib, pandas |
| **📝 Summarization** | Extractive (TextRank), abstractive (BART), key terms extraction, compression analysis | TextRank algorithm, facebook/bart-large-cnn, TF-IDF | NetworkX, transformers, scikit-learn, matplotlib |
| **📊 Topic Analysis** | LDA topic modeling, TF-IDF analysis, keyword extraction, topic classification | Latent Dirichlet Allocation, TF-IDF, NMF | scikit-learn, matplotlib, wordcloud, NLTK |
| **❓ Question Answering** | Context-based QA, answer extraction, confidence scoring, TF-IDF similarity | deepset/roberta-base-squad2, TF-IDF cosine similarity | transformers, scikit-learn, matplotlib, pandas |
| **✍️ Text Generation** | Text completion, creative writing, prompt-based generation, parameter control | GPT-2 (124M parameters) | transformers, matplotlib, pandas |
| **🌍 Translation** | Multi-language translation, language pair support, translation quality metrics | Helsinki-NLP/opus-mt models, MarianMT | transformers, langdetect, matplotlib |
| **📂 Classification** | Zero-shot classification, custom categories, multi-label support, confidence scoring | facebook/bart-large-mnli | transformers, matplotlib, pandas |
| **🔍 Vector Embeddings** | Semantic search, similarity analysis, embedding visualization, FAISS indexing | all-MiniLM-L6-v2, sentence-transformers | sentence-transformers, faiss, scikit-learn, matplotlib |

## 🔬 Upcoming Features

### 📊 **Mathematical Foundations**
- **Naive Bayes**: Step-by-step algorithmic breakdown with probability calculations
- **TF-IDF Deep Dive**: Vector space mathematics and similarity computations
- **Statistical NLP**: Probability distributions, language modeling, and statistical inference
- **Information Theory**: Entropy, mutual information, and perplexity calculations

### 🤖 **Advanced Algorithm Explanations**
- **Attention Mechanisms**: Mathematical formulation and visualization of attention weights
- **Transformer Architecture**: Layer-by-layer breakdown with matrix operations
- **Neural Language Models**: From n-grams to modern architectures
- **Optimization Techniques**: Gradient descent, backpropagation in NLP context

### 📈 **Enhanced Visualizations**
- **3D Embedding Spaces**: Interactive exploration of word and sentence embeddings
- **Algorithm Animations**: Real-time visualization of text processing steps
- **Parameter Tuning Interface**: Interactive model parameter exploration
- **Mathematical Plotting**: Dynamic visualization of statistical concepts

## 🛠️ Tools & Technologies

### 🧠 **NLP Models & Libraries**
- **Transformers**: GPT-2, BART, RoBERTa, BERT, DistilBERT (via transformers==4.36.2)
- **Traditional NLP**: spaCy (3.7.2), NLTK (3.8.1), TextBlob (0.17.1)
- **Embeddings**: Sentence Transformers (2.3.1), FAISS (1.7.4)
- **Translation**: Helsinki-NLP models, langdetect (1.0.9)

### 🖥️ **Backend & Infrastructure**
- **Framework**: Flask 3.0.0, flask-cors 4.0.0
- **Language**: Python 3.11
- **Containerization**: Docker
- **Deployment**: Hugging Face Spaces
- **Memory**: psutil (5.9.6) for optimization

### 📊 **Data Science & Visualization**
- **Analysis**: NumPy (1.24.3), Pandas (2.0.3), Scikit-learn (1.3.2)
- **Visualization**: Matplotlib (3.7.5), Seaborn (0.13.0), matplotlib-venn (0.11.10)
- **Network Analysis**: NetworkX (3.2.1)
- **Statistical Tools**: SciPy (1.11.4)
- **Text Visualization**: WordCloud (1.9.3)
- **Image Processing**: Pillow (10.1.0)

### 🎨 **Frontend**
- **Styling**: Bootstrap 5.3.0, Custom CSS
- **Interactivity**: Vanilla JavaScript
- **Charts**: Chart.js
- **Icons**: Font Awesome

## 🚀 Getting Started

### 📋 Prerequisites
- Python 3.11+
- Git
- Docker (optional)

### 🔧 Installation & Setup

#### Method 1: Standard Installation

1. **Clone the repository**
```bash
git clone https://github.com/aradhyapavan/NLP_Academy.git
cd NLP_Academy
```

2. **Create virtual environment**
```bash
python -m venv nlp_env
source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

5. **Run the application**
```bash
python app.py
```

6. **Access the application**
```
Open http://localhost:7860 in your browser
```

#### Method 2: Docker Installation

1. **Clone the repository**
```bash
git clone https://github.com/aradhyapavan/NLP_Academy.git
cd NLP_Academy
```

2. **Build Docker image**
```bash
docker build -t nlp-academy .
```

3. **Run Docker container**
```bash
docker run -p 7860:7860 nlp-academy
```

4. **Access the application**
```
Open http://localhost:7860 in your browser
```

#### Method 3: Docker Compose (Recommended)

1. **Clone and run**
```bash
git clone https://github.com/aradhyapavan/NLP_Academy.git
cd NLP_Academy
docker-compose up --build
```

## 🌐 Deployment

### Hugging Face Spaces
The application is deployed and accessible at:
**https://huggingface.co/spaces/aradhyapavan/nlp-ultimate-tutor**

### Local Development
```bash
# Development mode with auto-reload
export FLASK_ENV=development
python app.py
```

## 🎯 Usage Examples

### Basic Text Analysis
```python
# Example: Sentiment Analysis
text = "I love this NLP tutorial!"
# Navigate to /sentiment in the web interface
# Results: Positive sentiment with confidence scores
```

### Advanced Features
```python
# Example: Question Answering
context = "NLP Academy is an educational platform for learning NLP concepts."
question = "What is NLP Academy?"
# Use the /question-answering interface
# Results: Extracted answer with confidence metrics
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### 📝 Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include tests for new features
- Update documentation as needed

## 📊 Performance & Optimization

- **Memory Management**: Intelligent model caching and cleanup
- **CPU Optimization**: Efficient inference on CPU-only environments
- **Lazy Loading**: Models load on-demand to reduce startup time
- **Concurrent Users**: Optimized for multiple simultaneous users

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Download required models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('all')"
```

**Memory Issues**
```bash
# Reduce model cache size in utils/model_loader_hf.py
# Adjust max_models parameter in ModelCache class
```

**Port Conflicts**
```bash
# Change port in app.py or use environment variable
export PORT=5000
python app.py
```

## 📸 Snapshots

Below are screenshots of the app. Each title describes what the page shows.

**Home**

![NLP Ultimate Tutorial – Home](snapshots/1.NLP-Ultimate-Tutorial-Home.png)

**Text Preprocessing**

![Text Preprocessing](snapshots/2.Text-Preprocessing.png)

**Tokenization**

![Tokenization](snapshots/3.Tokenization.png)

**Part-of-Speech Tagging**

![Part-of-Speech Tagging](snapshots/4.Part-of-Speech-Tagging.png)

**Named Entity Recognition**

![Named Entity Recognition](snapshots/5.Named-Entity-Recognition.png)

**Sentiment Analysis**

![Sentiment Analysis](snapshots/6.Sentiment-Analysis.png)

**Text Summarization**

![Text Summarization](snapshots/7.Text Summarization.png)

**Topic Analysis**

![Topic Analysis](snapshots/8.Topic-Analysis.png)

**Question Answering**

![Question Answering](snapshots/9.Question-Answering.png)

**Text Generation**

![Text Generation](snapshots/10.Text-Generation.png)

**Machine Translation**

![Machine Translation](snapshots/11.Machine-Translation.png)

**Zero-shot Classification**

![Zero-shot Classification](snapshots/12.Zero-shot-Classification.png)

**Vector Embeddings & Semantic Search**

![Vector Embeddings & Semantic Search](snapshots/13.Vector-Embeddings-and-Semantic-Search.png)

## 📄 License

This project is licensed under the MIT License 

## 🙏 Acknowledgments

- **Hugging Face** for model hosting and deployment platform
- **BITS Pilani** for academic foundation
- **Open Source Community** for amazing libraries and tools
- **Contributors** who help improve this educational resource

## 📞 Contact & Support

- **GitHub**: [@aradhyapavan](https://github.com/aradhyapavan)
- **Project Link**: [https://github.com/aradhyapavan/NLP_Academy](https://github.com/aradhyapavan/NLP_Academy)
- **Live Demo**: [https://huggingface.co/spaces/aradhyapavan/nlp-ultimate-tutor](https://huggingface.co/spaces/aradhyapavan/nlp-ultimate-tutor)

---

⭐ **Star this repository if you find it helpful!** ⭐

*Built with ❤️ for the NLP learning community*
