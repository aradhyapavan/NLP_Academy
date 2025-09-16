"""
Optimized model loader for Hugging Face Spaces with memory management
"""
import os
import gc
import psutil
import nltk
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")

# Set device to CPU for HF Spaces (unless GPU is available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model cache with memory-conscious loading
class ModelCache:
    def __init__(self, max_models=3):
        self.models = {}
        self.max_models = max_models
        self.access_count = {}
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def cleanup_least_used(self):
        """Remove least recently used model if cache is full"""
        if len(self.models) >= self.max_models:
            # Find least used model
            least_used = min(self.access_count.items(), key=lambda x: x[1])
            model_name = least_used[0]
            
            print(f"Removing {model_name} from cache to free memory")
            del self.models[model_name]
            del self.access_count[model_name]
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def load_model(self, model_name, loader_func):
        """Load model with caching and memory management"""
        if model_name in self.models:
            self.access_count[model_name] += 1
            return self.models[model_name]
        
        # Check memory before loading
        memory_before = self.get_memory_usage()
        print(f"Memory before loading {model_name}: {memory_before:.1f}MB")
        
        # Clean up if necessary
        self.cleanup_least_used()
        
        # Load the model
        try:
            model = loader_func()
            self.models[model_name] = model
            self.access_count[model_name] = 1
            
            memory_after = self.get_memory_usage()
            print(f"Memory after loading {model_name}: {memory_after:.1f}MB")
            
            return model
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            return None

# Global model cache
model_cache = ModelCache(max_models=3)

@lru_cache(maxsize=1)
def download_nltk_resources():
    """Download and cache NLTK resources"""
    resources = ['punkt', 'stopwords', 'vader_lexicon', 'wordnet', 'averaged_perceptron_tagger']
    
    for resource in resources:
        try:
            if resource == 'punkt':
                nltk.data.find(f'tokenizers/{resource}')
            elif resource in ['stopwords', 'wordnet']:
                nltk.data.find(f'corpora/{resource}')
            elif resource == 'vader_lexicon':
                nltk.data.find(f'sentiment/{resource}')
            elif resource == 'averaged_perceptron_tagger':
                nltk.data.find(f'taggers/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)

@lru_cache(maxsize=1)
def load_spacy():
    """Load spaCy model with caching"""
    def _load_spacy():
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Please install: python -m spacy download en_core_web_sm")
            return None
    
    return model_cache.load_model("spacy", _load_spacy)

def load_sentiment_analyzer():
    """Load lightweight sentiment analyzer"""
    def _load_sentiment():
        return pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if DEVICE == "cuda" else -1,
            max_length=512,
            truncation=True
        )
    
    return model_cache.load_model("sentiment", _load_sentiment)

def load_summarizer():
    """Load efficient summarization model"""
    def _load_summarizer():
        return pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if DEVICE == "cuda" else -1,
            max_length=512,
            truncation=True
        )
    
    return model_cache.load_model("summarizer", _load_summarizer)

def load_qa_pipeline():
    """Load question-answering pipeline"""
    def _load_qa():
        return pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if DEVICE == "cuda" else -1,
            max_length=512,
            truncation=True
        )
    
    return model_cache.load_model("qa", _load_qa)

def load_text_generator():
    """Load text generation model"""
    def _load_generator():
        return pipeline(
            "text-generation",
            model="gpt2",
            device=0 if DEVICE == "cuda" else -1,
            max_length=256,
            truncation=True,
            pad_token_id=50256
        )
    
    return model_cache.load_model("generator", _load_generator)

def load_zero_shot():
    """Load zero-shot classification model"""
    def _load_zero_shot():
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if DEVICE == "cuda" else -1,
            max_length=512,
            truncation=True
        )
    
    return model_cache.load_model("zero_shot", _load_zero_shot)

def load_embedding_model():
    """Load sentence embedding model"""
    def _load_embedding():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    return model_cache.load_model("embedding", _load_embedding)

def load_translation_pipeline(source_lang="auto", target_lang="en"):
    """Load translation model with fallback"""
    def _load_translation():
        try:
            if source_lang == "auto" or target_lang == "en":
                model_name = "Helsinki-NLP/opus-mt-mul-en"
            else:
                model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            
            return pipeline(
                "translation",
                model=model_name,
                device=0 if DEVICE == "cuda" else -1,
                max_length=512,
                truncation=True
            )
        except Exception as e:
            print(f"Translation model error: {e}")
            return None
    
    return model_cache.load_model(f"translation_{source_lang}_{target_lang}", _load_translation)

def get_memory_status():
    """Get current memory usage statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent(),
        "loaded_models": list(model_cache.models.keys()),
        "cache_size": len(model_cache.models)
    }

def clear_model_cache():
    """Clear all models from cache to free memory"""
    model_cache.models.clear()
    model_cache.access_count.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model cache cleared")

def initialize_essential_models():
    """Initialize only the most essential models for startup"""
    print("Initializing essential models for Hugging Face Spaces...")
    
    # Download NLTK resources
    download_nltk_resources()
    print("✓ NLTK resources downloaded")
    
    # Load spaCy (small footprint)
    try:
        load_spacy()
        print("✓ spaCy model loaded")
    except Exception as e:
        print(f"✗ spaCy failed: {e}")
    
    # Load sentiment analyzer (most commonly used)
    try:
        load_sentiment_analyzer()
        print("✓ Sentiment analyzer loaded")
    except Exception as e:
        print(f"✗ Sentiment analyzer failed: {e}")
    
    print(f"Memory status: {get_memory_status()}")
    print("Essential models initialized!")

# Lazy loading functions for other models
def ensure_model_loaded(model_name, loader_func):
    """Ensure a model is loaded before use"""
    if model_name not in model_cache.models:
        print(f"Loading {model_name} on demand...")
        loader_func()
    return model_cache.models.get(model_name)

# Model status for debugging
def get_model_status():
    """Get status of all models"""
    return {
        "loaded_models": list(model_cache.models.keys()),
        "access_counts": model_cache.access_count.copy(),
        "memory_usage": get_memory_status(),
        "device": DEVICE
    }
