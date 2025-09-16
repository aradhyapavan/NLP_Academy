import nltk
import spacy
from transformers import pipeline

# Global models dictionary for persistent access
models = {
    "nlp": None,
    "sentiment_analyzer": None,
    "emotion_classifier": None,
    "summarizer": None,
    "qa_pipeline": None,
    "translation_pipeline": None,
    "text_generator": None,
    "zero_shot": None,
    "embedding_model": None
}

def download_nltk_resources():
    """Download and initialize NLTK resources"""
    resources = ['punkt', 'stopwords', 'vader_lexicon', 'wordnet', 'averaged_perceptron_tagger', 'sentiwordnet']
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
            elif resource == 'sentiwordnet':
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading required NLTK resource: {resource}")
            nltk.download(resource)

def load_spacy():
    """Load spaCy model"""
    if models["nlp"] is None:
        try:
            models["nlp"] = spacy.load("en_core_web_sm")
        except:
            print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    return models["nlp"]

def load_sentiment_analyzer():
    """Load sentiment analysis model"""
    if models["sentiment_analyzer"] is None:
        try:
            models["sentiment_analyzer"] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Failed to load sentiment analyzer: {e}")
    return models["sentiment_analyzer"]

def load_emotion_classifier():
    """Load emotion classification model"""
    if models["emotion_classifier"] is None:
        try:
            models["emotion_classifier"] = pipeline(
                "text-classification", 
                model="cardiffnlp/twitter-roberta-base-emotion",
                return_all_scores=True
            )
        except Exception as e:
            print(f"Failed to load emotion classifier: {e}")
    return models["emotion_classifier"]

def load_summarizer():
    """Load summarization model"""
    if models["summarizer"] is None:
        try:
            models["summarizer"] = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            print(f"Failed to load summarizer: {e}")
    return models["summarizer"]

def load_qa_pipeline():
    """Load or initialize the question answering pipeline."""
    if models["qa_pipeline"] is None:
        try:
            from transformers import pipeline
            
            # Use a smaller model to reduce memory usage and improve speed
            models["qa_pipeline"] = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",  # You can change this to a different model if needed
                tokenizer="deepset/roberta-base-squad2"
            )
        except Exception as e:
            print(f"Error loading QA pipeline: {e}")
            models["qa_pipeline"] = None
            raise e
    return models["qa_pipeline"]

def load_translation_pipeline():
    """Load translation model"""
    if models["translation_pipeline"] is None:
        try:
            models["translation_pipeline"] = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
        except Exception as e:
            print(f"Failed to load translation model: {e}")
    return models["translation_pipeline"]

def load_translator(source_lang="auto", target_lang="en"):
    """
    Load a machine translation model for the given language pair.
    
    Args:
        source_lang (str): Source language code, or 'auto' for automatic detection
        target_lang (str): Target language code
        
    Returns:
        A translation pipeline or model
    """
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    
    try:
        # For auto language detection, use a more general model
        if source_lang == "auto":
            # Using Helsinki-NLP's opus-mt model for translation
            model_name = "Helsinki-NLP/opus-mt-mul-en"  # Multilingual to English
            translator = pipeline("translation", model=model_name)
        else:
            # For specific language pairs
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            
            # Load the model and tokenizer
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create the translation pipeline
            translator = pipeline("translation", model=model, tokenizer=tokenizer)
            
        return translator
    except Exception as e:
        # Fallback to a more general model if language pair isn't available
        try:
            # Use MarianMT model for many language pairs
            model_name = "Helsinki-NLP/opus-mt-mul-en"  # Multilingual to English
            translator = pipeline("translation", model=model_name)
            return translator
        except Exception as nested_e:
            # If all else fails, return a simple callable object that returns an error message
            class ErrorTranslator:
                def __call__(self, text, **kwargs):
                    return [{"translation_text": f"Error loading translation model: {str(e)}. Fallback also failed: {str(nested_e)}"}]
            return ErrorTranslator()

def load_text_generator():
    """Load text generation model"""
    if models["text_generator"] is None:
        try:
            models["text_generator"] = pipeline("text-generation", model="gpt2")
        except Exception as e:
            print(f"Failed to load text generator: {e}")
    return models["text_generator"]

def load_zero_shot():
    """Load zero-shot classification model"""
    if models["zero_shot"] is None:
        try:
            models["zero_shot"] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            print(f"Failed to load zero-shot classifier: {e}")
    return models["zero_shot"]

def load_embedding_model():
    """Load sentence embedding model for semantic search"""
    if models.get("embedding_model") is None:
        try:
            from sentence_transformers import SentenceTransformer
            models["embedding_model"] = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
    return models["embedding_model"]

def initialize_all_models():
    """Initialize all models for better performance"""
    print("Initializing NLP models...")
    
    # Download NLTK resources first
    download_nltk_resources()
    
    # Load spaCy model
    try:
        load_spacy()
        print("✓ spaCy model loaded")
    except Exception as e:
        print(f"✗ Failed to load spaCy: {e}")
    
    # Load transformer models (these might take time)
    models_to_load = [
        ("Sentiment Analyzer", load_sentiment_analyzer),
        ("Emotion Classifier", load_emotion_classifier),
        ("Summarizer", load_summarizer),
        ("QA Pipeline", load_qa_pipeline),
        ("Text Generator", load_text_generator),
        ("Zero-shot Classifier", load_zero_shot),
        ("Embedding Model", load_embedding_model)
    ]
    
    for name, loader_func in models_to_load:
        try:
            loader_func()
            print(f"✓ {name} loaded")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    print("Model initialization complete!")

def get_model_status():
    """Get status of all models"""
    status = {}
    for model_name, model in models.items():
        status[model_name] = model is not None
    return status

def clear_models():
    """Clear all loaded models to free memory"""
    for key in models:
        models[key] = None
    print("All models cleared from memory")
