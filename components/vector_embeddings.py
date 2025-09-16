import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import time
import faiss
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import textwrap
from sklearn.metrics.pairwise import cosine_similarity

from utils.model_loader import load_embedding_model
from utils.helpers import fig_to_html, df_to_html_table

def vector_embeddings_handler(text_input, search_query=""):
    """Show vector embeddings and semantic search capabilities."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Vector Embeddings Analysis Results</h2>')
    
    output_html.append("""
    <div class="alert alert-success">
        <h4><i class="fas fa-check-circle me-2"></i>Embeddings Generated Successfully!</h4>
        <p class="mb-0">Your text has been processed and converted into high-dimensional vector representations.</p>
    </div>
    """)
    
    # Load model and create embeddings
    try:
        model = load_embedding_model()
        
        # Split the text into chunks (sentences)
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text_input)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        # If we have too few sentences, create artificial chunks
        if len(sentences) < 3:
            words = text_input.split()
            chunk_size = max(10, len(words) // 3)
            sentences = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size) if i+chunk_size <= len(words)]
        
        # Limit to 10 sentences to avoid overwhelming the visualization
        if len(sentences) > 10:
            sentences = sentences[:10]
        
        # Create embeddings
        embeddings = model.encode(sentences)
        
        # Text Statistics
        output_html.append(f"""
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0"><i class="fas fa-chart-bar me-2"></i>Processing Statistics</h4>
                    </div>
                    <div class="card-body">
                        <div class="row text-center">
                            <div class="col-md-3">
                                <div class="stat-item">
                                    <h3 class="text-primary">{len(text_input)}</h3>
                                    <p class="text-muted mb-0">Characters</p>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="stat-item">
                                    <h3 class="text-success">{len(sentences)}</h3>
                                    <p class="text-muted mb-0">Text Segments</p>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="stat-item">
                                    <h3 class="text-info">{embeddings.shape[1]}</h3>
                                    <p class="text-muted mb-0">Vector Dimensions</p>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="stat-item">
                                    <h3 class="text-warning">{embeddings.shape[0]}</h3>
                                    <p class="text-muted mb-0">Embedding Vectors</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # Text Segments Display
        output_html.append("""
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-info text-white">
                        <h4 class="mb-0"><i class="fas fa-list me-2"></i>Text Segments</h4>
                    </div>
                    <div class="card-body">
                        <div class="row">
        """)
        
        for i, sentence in enumerate(sentences[:6]):  # Show max 6 segments
            output_html.append(f"""
                            <div class="col-md-6 mb-3">
                                <div class="p-3 border rounded bg-light">
                                    <h6 class="text-primary mb-2">Segment {i+1}</h6>
                                    <p class="mb-0 small">{sentence}</p>
                                </div>
                            </div>
            """)
        
        output_html.append("""
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # Semantic Search Interface
        output_html.append("""
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-warning">
                    <div class="card-header bg-warning text-dark">
                        <h4 class="mb-0"><i class="fas fa-search me-2"></i>Semantic Search</h4>
                    </div>
                    <div class="card-body">
                        <p class="mb-3">Search for content by meaning, not just keywords. The system will find the most semantically similar text segments.</p>
                        
                        <div class="row mb-3">
                            <div class="col-md-10">
                                <input type="text" id="search-input" class="form-control form-control-lg" placeholder="Enter a search query to find similar content...">
                            </div>
                            <div class="col-md-2">
                                <button onclick="performSemanticSearch()" class="btn btn-warning btn-lg w-100">
                                    <i class="fas fa-search me-1"></i>Search
                                </button>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <h6 class="mb-2"><i class="fas fa-lightbulb me-2"></i>Try these example searches:</h6>
                            <div class="d-flex flex-wrap gap-2">
                                <button onclick="document.getElementById('search-input').value = 'space research'; performSemanticSearch();" 
                                        class="btn btn-outline-secondary btn-sm">
                                    <i class="fas fa-rocket me-1"></i>space research
                                </button>
                                <button onclick="document.getElementById('search-input').value = 'scientific collaboration'; performSemanticSearch();" 
                                        class="btn btn-outline-secondary btn-sm">
                                    <i class="fas fa-users me-1"></i>scientific collaboration
                                </button>
                                <button onclick="document.getElementById('search-input').value = 'international project'; performSemanticSearch();" 
                                        class="btn btn-outline-secondary btn-sm">
                                    <i class="fas fa-globe me-1"></i>international project
                                </button>
                                <button onclick="document.getElementById('search-input').value = 'laboratory experiments'; performSemanticSearch();" 
                                        class="btn btn-outline-secondary btn-sm">
                                    <i class="fas fa-flask me-1"></i>laboratory experiments
                                </button>
                                <button onclick="document.getElementById('search-input').value = 'space agencies'; performSemanticSearch();" 
                                        class="btn btn-outline-secondary btn-sm">
                                    <i class="fas fa-building me-1"></i>space agencies
                                </button>
                                <button onclick="document.getElementById('search-input').value = 'microgravity environment'; performSemanticSearch();" 
                                        class="btn btn-outline-secondary btn-sm">
                                    <i class="fas fa-weight me-1"></i>microgravity environment
                                </button>
                            </div>
                        </div>
                        
                        <div id="search-results" style="display: none;">
                            <hr>
                            <h5><i class="fas fa-list-ol me-2"></i>Search Results:</h5>
                            <div id="results-container" class="border rounded p-3 bg-light" style="max-height: 400px; overflow-y: auto;">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """)
        
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h4><i class="fas fa-exclamation-triangle me-2"></i>Error</h4>
            <p>Could not generate embeddings: {str(e)}</p>
        </div>
        """)
    
    # Close result-area div
    output_html.append('</div>')
    return '\n'.join(output_html)

def perform_semantic_search(context, query):
    """Perform semantic search on the given context with the query."""
    try:
        # Load model
        model = load_embedding_model()
        
        # Split context into sentences
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(context)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        
        # Create embeddings
        sentence_embeddings = model.encode(sentences)
        query_embedding = model.encode([query])[0]
        
        # Calculate similarities
        from sentence_transformers import util
        similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0].cpu().numpy()
        
        # Create result pairs (sentence, similarity)
        results = [(sentences[i], float(similarities[i])) for i in range(len(sentences))]
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        return {
            "success": True,
            "results": [
                {"text": text, "score": score}
                for text, score in results[:5]  # Return top 5 results
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }