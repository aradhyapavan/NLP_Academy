import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from transformers import pipeline
import nltk
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.model_loader import load_qa_pipeline
from utils.helpers import fig_to_html, df_to_html_table

def question_answering_handler(context_text, question, answer_type="extractive", confidence_threshold=0.5):
    """Show question answering capabilities with comprehensive analysis."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Question Answering System</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-info-circle"></i>
    Question Answering (QA) systems extract or generate answers to questions based on a given context or knowledge base.
    This system can handle both extractive (finding answers in text) and abstractive (generating new answers) approaches.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Models & Techniques Used:</h4>
        <ul>
            <li><b>RoBERTa-SQuAD2</b> - Fine-tuned transformer model for extractive QA (F1: ~83.7 on SQuAD 2.0)</li>
            <li><b>BERT-based QA</b> - Bidirectional encoder representations for understanding context</li>
            <li><b>TF-IDF Similarity</b> - Traditional approach for finding relevant text spans</li>
            <li><b>Confidence Scoring</b> - Model uncertainty estimation for answer reliability</li>
        </ul>
    </div>
    """)
    
    try:
        # Validate inputs
        if not context_text or not context_text.strip():
            output_html.append('<div class="alert alert-warning">⚠️ Please provide a context text for question answering.</div>')
            output_html.append('</div>')
            return "\n".join(output_html)
            
        if not question or not question.strip():
            output_html.append('<div class="alert alert-warning">⚠️ Please provide a question to answer.</div>')
            output_html.append('</div>')
            return "\n".join(output_html)
        
        # Display input information
        output_html.append('<h3 class="task-subheader">Input Analysis</h3>')
        
        context_stats = {
            "Context Length": len(context_text),
            "Word Count": len(context_text.split()),
            "Sentence Count": len(nltk.sent_tokenize(context_text)),
            "Question Length": len(question),
            "Question Words": len(question.split())
        }
        
        stats_df = pd.DataFrame(list(context_stats.items()), columns=['Metric', 'Value'])
        output_html.append('<h4>Input Statistics</h4>')
        output_html.append(df_to_html_table(stats_df))
        
        # Question Analysis
        output_html.append('<h3 class="task-subheader">Question Analysis</h3>')
        
        # Classify question type
        question_lower = question.lower().strip()
        question_type = classify_question_type(question_lower)
        
        output_html.append(f"""
        <div class="card">
            <div class="card-header">
                <h4 class="mb-0">Question Classification</h4>
            </div>
            <div class="card-body">
                <p><strong>Question:</strong> {question}</p>
                <p><strong>Type:</strong> {question_type['type']}</p>
                <p><strong>Expected Answer:</strong> {question_type['expected']}</p>
                <p><strong>Keywords:</strong> {', '.join(question_type['keywords'])}</p>
            </div>
        </div>
        """)
        
        # Extractive Question Answering using Transformer
        output_html.append('<h3 class="task-subheader">Transformer-based Answer Extraction</h3>')
        
        try:
            qa_pipeline = load_qa_pipeline()
            
            # Get answer from the model
            result = qa_pipeline(question=question, context=context_text)
            
            answer = result['answer']
            confidence = result['score']
            start_pos = result['start']
            end_pos = result['end']
            
            # Create confidence visualization
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            
            # Confidence bar
            colors = ['red' if confidence < 0.3 else 'orange' if confidence < 0.7 else 'green']
            bars = ax.barh(['Confidence'], [confidence], color=colors[0])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Confidence Score')
            ax.set_title('Answer Confidence')
            
            # Add confidence threshold line
            ax.axvline(x=confidence_threshold, color='red', linestyle='--', label=f'Threshold ({confidence_threshold})')
            ax.legend()
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width/2, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='center', va='center', fontweight='bold')
            
            plt.tight_layout()
            output_html.append(fig_to_html(fig))
            plt.close()
            
            # Display answer with context highlighting
            confidence_status = "High" if confidence >= 0.7 else "Medium" if confidence >= 0.3 else "Low"
            confidence_color = "#4CAF50" if confidence >= 0.7 else "#FF9800" if confidence >= 0.3 else "#F44336"
            
            output_html.append(f"""
            <div class="card" style="border-color: {confidence_color};">
                <div class="card-header" style="background-color: {confidence_color}22;">
                    <h4 class="mb-0">📝 Extracted Answer</h4>
                </div>
                <div class="card-body">
                    <div class="alert alert-light">
                        <strong>Answer:</strong> <span class="badge bg-warning text-dark fs-6">{answer}</span>
                    </div>
                    <p><strong>Confidence:</strong> {confidence:.3f} ({confidence_status})</p>
                    <p><strong>Position in Text:</strong> Characters {start_pos}-{end_pos}</p>
                </div>
            </div>
            """)
            
            # Show context with answer highlighted
            highlighted_context = highlight_answer_in_context(context_text, start_pos, end_pos)
            output_html.append(f"""
            <div class="card">
                <div class="card-header">
                    <h4 class="mb-0">📄 Context with Highlighted Answer</h4>
                </div>
                <div class="card-body">
                    <div style="line-height: 1.6; border: 1px solid #ddd; padding: 1rem; border-radius: 5px;">
                        {highlighted_context}
                    </div>
                </div>
            </div>
            """)
            
        except Exception as e:
            output_html.append(f'<div class="alert alert-danger">❌ Error in transformer QA: {str(e)}</div>')
        
        # Alternative: TF-IDF based answer extraction
        output_html.append('<h3 class="task-subheader">TF-IDF Based Answer Extraction</h3>')
        
        try:
            tfidf_answer = extract_answer_tfidf(context_text, question)
            
            output_html.append(f"""
            <div class="alert alert-success">
                <h4>🔍 TF-IDF Based Answer</h4>
                <div class="alert alert-light">
                    <strong>Most Relevant Sentence:</strong> {tfidf_answer['sentence']}
                </div>
                <p><strong>Similarity Score:</strong> {tfidf_answer['score']:.3f}</p>
                <p><strong>Method:</strong> Cosine similarity between question and context sentences using TF-IDF vectors</p>
            </div>
            """)
            
        except Exception as e:
            output_html.append(f'<div class="alert alert-danger">❌ Error in TF-IDF QA: {str(e)}</div>')
        
        # Answer Quality Assessment
        output_html.append('<h3 class="task-subheader">Answer Quality Assessment</h3>')
        
        if 'confidence' in locals():
            quality_metrics = assess_answer_quality(question, answer, confidence, context_text)
            
            # Create quality assessment visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Quality metrics radar chart
            categories = list(quality_metrics.keys())
            values = list(quality_metrics.values())
            
            ax1.bar(categories, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
            ax1.set_ylim(0, 1)
            ax1.set_title('Answer Quality Metrics')
            ax1.set_ylabel('Score')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Overall quality score
            overall_score = sum(values) / len(values)
            quality_label = "Excellent" if overall_score >= 0.8 else "Good" if overall_score >= 0.6 else "Fair" if overall_score >= 0.4 else "Poor"
            
            ax2.pie([overall_score, 1-overall_score], labels=[f'{quality_label}\n({overall_score:.2f})', 'Room for Improvement'], 
                   colors=['#4CAF50', '#E0E0E0'], startangle=90)
            ax2.set_title('Overall Answer Quality')
            
            plt.tight_layout()
            output_html.append(fig_to_html(fig))
            plt.close()
            
            # Quality metrics table
            quality_df = pd.DataFrame([
                {'Metric': 'Confidence', 'Score': f"{quality_metrics['Confidence']:.3f}", 'Description': 'Model confidence in the answer'},
                {'Metric': 'Relevance', 'Score': f"{quality_metrics['Relevance']:.3f}", 'Description': 'Semantic similarity to question'},
                {'Metric': 'Completeness', 'Score': f"{quality_metrics['Completeness']:.3f}", 'Description': 'Answer length appropriateness'},
                {'Metric': 'Context Match', 'Score': f"{quality_metrics['Context_Match']:.3f}", 'Description': 'How well answer fits context'}
            ])
            
            output_html.append('<h4>Quality Assessment Details</h4>')
            output_html.append(df_to_html_table(quality_df))
        
        # Question-Answer Pairs Suggestions
        output_html.append('<h3 class="task-subheader">Suggested Follow-up Questions</h3>')
        
        try:
            suggested_questions = generate_followup_questions(context_text, question, answer if 'answer' in locals() else "")
            
            output_html.append('<div class="alert alert-warning">')
            output_html.append('<h4>💡 Follow-up Questions:</h4>')
            output_html.append('<ul>')
            for i, q in enumerate(suggested_questions, 1):
                output_html.append(f'<li><strong>Q{i}:</strong> {q}</li>')
            output_html.append('</ul>')
            output_html.append('</div>')
            
        except Exception as e:
            output_html.append(f'<div class="alert alert-danger">❌ Error generating suggestions: {str(e)}</div>')
        
    except Exception as e:
        output_html.append(f'<div class="alert alert-danger">❌ Unexpected error: {str(e)}</div>')
    
    output_html.append('</div>')
    return "\n".join(output_html)

def classify_question_type(question):
    """Classify the type of question and expected answer format."""
    question = question.lower().strip()
    
    # Question word patterns
    patterns = {
        'what': {'type': 'Definition/Fact', 'expected': 'Entity, concept, or description'},
        'who': {'type': 'Person', 'expected': 'Person name or group'},
        'when': {'type': 'Time', 'expected': 'Date, time, or temporal expression'},
        'where': {'type': 'Location', 'expected': 'Place, location, or spatial reference'},
        'why': {'type': 'Reason/Cause', 'expected': 'Explanation or causal relationship'},
        'how': {'type': 'Method/Process', 'expected': 'Process, method, or manner'},
        'which': {'type': 'Selection', 'expected': 'Specific choice from options'},
        'how much': {'type': 'Quantity', 'expected': 'Numerical amount or quantity'},
        'how many': {'type': 'Count', 'expected': 'Numerical count'},
        'is': {'type': 'Yes/No', 'expected': 'Boolean answer'},
        'are': {'type': 'Yes/No', 'expected': 'Boolean answer'},
        'can': {'type': 'Ability/Possibility', 'expected': 'Yes/No with explanation'},
        'will': {'type': 'Future/Prediction', 'expected': 'Future state or prediction'},
        'did': {'type': 'Past Action', 'expected': 'Yes/No about past events'}
    }
    
    # Extract keywords from question
    words = question.split()
    keywords = [word for word in words if len(word) > 2 and word not in ['the', 'and', 'but', 'for']]
    
    # Determine question type
    for pattern, info in patterns.items():
        if question.startswith(pattern):
            return {
                'type': info['type'],
                'expected': info['expected'],
                'keywords': keywords[:5]  # Top 5 keywords
            }
    
    # Default classification
    return {
        'type': 'General',
        'expected': 'Text span or explanation',
        'keywords': keywords[:5]
    }

def extract_answer_tfidf(context, question):
    """Extract answer using TF-IDF similarity."""
    # Split context into sentences
    sentences = nltk.sent_tokenize(context)
    
    if len(sentences) == 0:
        return {'sentence': 'No sentences found', 'score': 0.0}
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    
    # Combine question with sentences for vectorization
    texts = [question] + sentences
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate cosine similarity between question and each sentence
    question_vector = tfidf_matrix[0:1]
    sentence_vectors = tfidf_matrix[1:]
    
    similarities = cosine_similarity(question_vector, sentence_vectors).flatten()
    
    # Find the most similar sentence
    best_idx = np.argmax(similarities)
    best_sentence = sentences[best_idx]
    best_score = similarities[best_idx]
    
    return {
        'sentence': best_sentence,
        'score': best_score
    }

def highlight_answer_in_context(context, start_pos, end_pos):
    """Highlight the answer span in the context."""
    before = context[:start_pos]
    answer = context[start_pos:end_pos]
    after = context[end_pos:]
    
    highlighted = f'{before}<mark style="background-color: #FFEB3B; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{answer}</mark>{after}'
    
    return highlighted

def assess_answer_quality(question, answer, confidence, context):
    """Assess the quality of the extracted answer."""
    metrics = {}
    
    # Confidence score (from model)
    metrics['Confidence'] = confidence
    
    # Relevance (simple keyword overlap)
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(question_words.intersection(answer_words))
    metrics['Relevance'] = min(overlap / max(len(question_words), 1), 1.0)
    
    # Completeness (answer length appropriateness)
    answer_length = len(answer.split())
    if answer_length == 0:
        metrics['Completeness'] = 0.0
    elif answer_length < 3:
        metrics['Completeness'] = 0.6
    elif answer_length <= 20:
        metrics['Completeness'] = 1.0
    else:
        metrics['Completeness'] = 0.8  # Very long answers might be too verbose
    
    # Context match (how well the answer fits in context)
    answer_in_context = answer.lower() in context.lower()
    metrics['Context_Match'] = 1.0 if answer_in_context else 0.5
    
    return metrics

def generate_followup_questions(context, original_question, answer):
    """Generate relevant follow-up questions based on the context and answer."""
    suggestions = []
    
    # Extract key entities and concepts from context
    words = context.split()
    
    # Template-based question generation
    templates = [
        f"What else can you tell me about {answer}?",
        "Can you provide more details about this topic?",
        "What are the implications of this information?",
        "How does this relate to other concepts mentioned?",
        "What evidence supports this answer?"
    ]
    
    # Add context-specific questions
    if "when" not in original_question.lower():
        suggestions.append("When did this happen?")
    
    if "where" not in original_question.lower():
        suggestions.append("Where did this take place?")
    
    if "why" not in original_question.lower():
        suggestions.append("Why is this significant?")
    
    if "how" not in original_question.lower():
        suggestions.append("How does this work?")
    
    # Combine and limit suggestions
    all_suggestions = templates + suggestions
    return all_suggestions[:5]  # Return top 5 suggestions

def qa_api_handler(context, question):
    """API handler for question answering that returns structured data."""
    try:
        qa_pipeline = load_qa_pipeline()
        result = qa_pipeline(question=question, context=context)
        
        return {
            "answer": result['answer'],
            "confidence": result['score'],
            "start_position": result['start'],
            "end_position": result['end'],
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "answer": "",
            "confidence": 0.0,
            "start_position": 0,
            "end_position": 0,
            "success": False,
            "error": str(e)
        }

def process_question_with_context(context_text, question):
    """Process a question with the given context and return a formatted result."""
    if not context_text or not context_text.strip():
        return {
            "success": False,
            "error": "No context text provided",
            "html": '<div class="alert alert-warning">⚠️ No context text provided.</div>'
        }
    
    if not question or not question.strip():
        return {
            "success": False,
            "error": "No question provided",
            "html": '<div class="alert alert-warning">⚠️ Please enter a question.</div>'
        }
    
    try:
        qa_pipeline = load_qa_pipeline()
        result = qa_pipeline(question=question, context=context_text)
        
        answer = result['answer']
        confidence = result['score']
        start_pos = result['start']
        end_pos = result['end']
        
        # Determine confidence level
        confidence_status = "High" if confidence >= 0.7 else "Medium" if confidence >= 0.3 else "Low"
        confidence_color = "#4CAF50" if confidence >= 0.7 else "#FF9800" if confidence >= 0.3 else "#F44336"
        
        # Highlight answer in context
        highlighted_context = highlight_answer_in_context(context_text, start_pos, end_pos)
        
        # Create formatted HTML result
        html_result = f"""
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">📝 Answer Found!</h5>
            </div>
            <div class="card-body">
                <div class="alert alert-light">
                    <p><strong>Question:</strong> {question}</p>
                    <p><strong>Answer:</strong> <span class="badge bg-warning text-dark fs-6">{answer}</span></p>
                    <p><strong>Confidence:</strong> {confidence:.3f} ({confidence_status})</p>
                </div>
                
                <div class="alert alert-light">
                    <h6>📄 Context with Highlighted Answer:</h6>
                    <div style="line-height: 1.6; font-size: 0.9rem; max-height: 200px; overflow-y: auto;">
                        {highlighted_context}
                    </div>
                </div>
                
                <div class="alert alert-info">
                    <strong>Quality Assessment:</strong>
                    <ul class="mb-0">
                        <li>Confidence: {confidence_status} ({confidence:.1%})</li>
                        <li>Answer found at position: {start_pos}-{end_pos}</li>
                        <li>Answer length: {len(answer.split())} words</li>
                    </ul>
                </div>
            </div>
        </div>
        """
        
        return {
            "success": True,
            "answer": answer,
            "confidence": confidence,
            "html": html_result
        }
        
    except Exception as e:
        error_html = f'<div class="alert alert-danger">❌ Error processing question: {str(e)}</div>'
        return {
            "success": False,
            "error": str(e),
            "html": error_html
        }
