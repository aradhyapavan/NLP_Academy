from flask import Flask, render_template, request, jsonify, session
import os
import json
from datetime import datetime

# Import components
from components.preprocessing import preprocessing_handler
from components.tokenization import tokenization_handler
from components.pos_tagging import pos_tagging_handler
from components.named_entity import named_entity_handler
from components.sentiment import sentiment_handler
from components.summarization import summarization_handler
from components.topic_analysis import topic_analysis_handler
from components.question_answering import question_answering_handler
from components.text_generation import text_generation_handler
from components.translation import translation_handler
from components.classification import classification_handler
from components.vector_embeddings import vector_embeddings_handler

# Import utilities
from utils.model_loader_hf import download_nltk_resources, load_spacy, initialize_essential_models
from utils.helpers import text_statistics

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Sample texts
SAMPLE_TEXTS = {
    "News Article": "The European Commission has fined Google €1.49 billion for abusive practices in online advertising. Google abused its market dominance by imposing restrictive clauses in contracts with third-party websites, preventing competitors from placing their search adverts on these websites.",
    "Product Review": "I absolutely love this smartphone! The camera quality is outstanding and the battery life is impressive. The user interface is intuitive and the performance is smooth even when running multiple apps. However, I find the price a bit high compared to similar models on the market.",
    "Scientific Text": "Climate change is the long-term alteration of temperature and typical weather patterns in a place. The cause of current climate change is largely human activity, like burning fossil fuels, which adds heat-trapping gases to Earth's atmosphere. The consequences of changing climate are already being felt worldwide.",
    "Literary Text": "The old man was thin and gaunt with deep wrinkles in the back of his neck. The brown blotches of the benevolent skin cancer the sun brings from its reflection on the tropical sea were on his cheeks. The blotches ran well down the sides of his face and his hands had the deep-creased scars from handling heavy fish on the cords."
}

# Initialize essential models for HF Spaces
initialize_essential_models()

@app.route('/')
def index():
    """Main page with text input and analysis options"""
    return render_template('index.html', sample_texts=SAMPLE_TEXTS)

@app.route('/api/text-stats', methods=['POST'])
def get_text_stats():
    """API endpoint to get text statistics"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    stats = text_statistics(text)
    return jsonify(stats)

@app.route('/api/sample-text', methods=['POST'])
def get_sample_text():
    """API endpoint to get sample text"""
    data = request.get_json()
    sample_type = data.get('sample_type', 'Custom')
    
    if sample_type == "Custom":
        return jsonify({'text': ''})
    else:
        return jsonify({'text': SAMPLE_TEXTS.get(sample_type, '')})

# Text Processing Routes
@app.route('/preprocessing')
def preprocessing():
    """Text preprocessing page"""
    return render_template('preprocessing.html')

@app.route('/api/preprocessing', methods=['POST'])
def api_preprocessing():
    """API endpoint for text preprocessing"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = preprocessing_handler(text)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tokenization')
def tokenization():
    """Tokenization page"""
    return render_template('tokenization.html')

@app.route('/api/tokenization', methods=['POST'])
def api_tokenization():
    """API endpoint for tokenization"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = tokenization_handler(text)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pos-tagging')
def pos_tagging():
    """POS tagging page"""
    return render_template('pos_tagging.html')

@app.route('/api/pos-tagging', methods=['POST'])
def api_pos_tagging():
    """API endpoint for POS tagging"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = pos_tagging_handler(text)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/named-entity')
def named_entity():
    """Named entity recognition page"""
    return render_template('named_entity.html')

@app.route('/api/named-entity', methods=['POST'])
def api_named_entity():
    """API endpoint for named entity recognition"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = named_entity_handler(text)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Analysis Routes
@app.route('/sentiment')
def sentiment():
    """Sentiment analysis page"""
    return render_template('sentiment.html')

@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    """API endpoint for sentiment analysis"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = sentiment_handler(text)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarization')
def summarization():
    """Text summarization page"""
    return render_template('summarization.html')

@app.route('/api/summarization', methods=['POST'])
def api_summarization():
    """API endpoint for text summarization"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = summarization_handler(text)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/topic-analysis')
def topic_analysis():
    """Topic analysis page"""
    return render_template('topic_analysis.html')

@app.route('/api/topic-analysis', methods=['POST'])
def api_topic_analysis():
    """API endpoint for topic analysis"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = topic_analysis_handler(text)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Advanced NLP Routes
@app.route('/question-answering')
def question_answering():
    """Question answering page"""
    return render_template('question_answering.html')

@app.route('/api/question-answering', methods=['POST'])
def api_question_answering():
    """API endpoint for question answering"""
    data = request.get_json(silent=True) or {}
    # Accept from JSON, form, or query string
    text = (
        data.get('context')
        or data.get('text')
        or request.form.get('context')
        or request.form.get('text')
        or request.args.get('context')
        or request.args.get('text')
        or ''
    )
    question = (
        data.get('question')
        or request.form.get('question')
        or request.args.get('question')
        or ''
    )
    confidence_threshold = (
        data.get('confidence_threshold')
        or request.form.get('confidence_threshold')
        or request.args.get('confidence_threshold')
        or 0.5
    )
    try:
        confidence_threshold = float(confidence_threshold)
    except Exception:
        confidence_threshold = 0.5
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = question_answering_handler(text, question, confidence_threshold=confidence_threshold)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/text-generation')
def text_generation():
    """Text generation page"""
    return render_template('text_generation.html')

@app.route('/api/text-generation', methods=['POST'])
def api_text_generation():
    """API endpoint for text generation"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = text_generation_handler(text)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translation')
def translation():
    """Translation page"""
    return render_template('translation.html')

@app.route('/api/translation', methods=['POST'])
def api_translation():
    """API endpoint for translation"""
    data = request.get_json()
    text = data.get('text', '')
    target_language = data.get('target_language', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = translation_handler(text, target_language)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classification')
def classification():
    """Classification page"""
    return render_template('classification.html')

@app.route('/api/classification', methods=['POST'])
def api_classification():
    """API endpoint for classification"""
    data = request.get_json()
    text = data.get('text', '')
    scenario = data.get('scenario', 'Sentiment')
    multi_label = data.get('multi_label', False)
    custom_labels = data.get('custom_labels', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = classification_handler(text, scenario, multi_label, custom_labels)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/vector-embeddings')
def vector_embeddings():
    """Vector embeddings page"""
    return render_template('vector_embeddings.html')

@app.route('/api/vector-embeddings', methods=['POST'])
def api_vector_embeddings():
    """API endpoint for vector embeddings"""
    data = request.get_json()
    text = data.get('text', '')
    query = data.get('query', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = vector_embeddings_handler(text, query)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/semantic-search', methods=['POST'])
def api_semantic_search():
    """API endpoint for semantic search"""
    from components.vector_embeddings import perform_semantic_search
    
    data = request.get_json()
    context = data.get('context', '')
    query = data.get('query', '')
    
    if not context or not query:
        return jsonify({'error': 'Both context and query are required'}), 400
    
    try:
        result = perform_semantic_search(context, query)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
