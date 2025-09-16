import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from collections import Counter
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib_venn import venn2

from utils.model_loader import load_summarizer
from utils.helpers import fig_to_html, df_to_html_table

def summarization_handler(text_input, min_length=30, max_length=300, use_sampling=False):
    """Show text summarization capabilities."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Text Summarization</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-info-circle"></i>
    Text summarization condenses text to capture its main points, enabling quicker comprehension of large volumes of information.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Models & Techniques Used:</h4>
        <ul>
            <li><b>Extractive Summarization</b> - Selects important sentences from the original text</li>
            <li><b>Abstractive Summarization</b> - BART model fine-tuned on CNN/DM dataset to generate new summary text</li>
            <li><b>Performance</b> - ROUGE scores of approximately 40-45 on CNN/DM benchmark</li>
        </ul>
    </div>
    """)
    
    try:
        # Check if text is long enough for summarization
        sentences = nltk.sent_tokenize(text_input)
        word_count = len(text_input.split())
        
        if len(sentences) < 3 or word_count < 40:
            output_html.append(f"""
            <div class="alert alert-warning">
                <h3>Text Too Short for Summarization</h3>
                <p>The provided text contains only {len(sentences)} sentences and {word_count} words. 
                For effective summarization, please provide a longer text (at least 3 sentences and 40 words).</p>
            </div>
            """)
        else:
            # Original Text Section
            output_html.append('<h3 class="task-subheader">Original Text</h3>')
            output_html.append(f"""
            <div class="card">
                <div class="card-body">
                    <div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 500px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">{text_input}</div>
                </div>
            </div>
            <p>Length: {word_count} words.</p>
            """)
            
            # Text Statistics
            char_count = len(text_input)
            avg_sentence_length = word_count / len(sentences)
            avg_word_length = sum(len(word) for word in text_input.split()) / word_count
            
            # Neural Summarization Section
            output_html.append('<h3 class="task-subheader">Neural Abstractive Summarization</h3>')
            output_html.append('<p>Using BART model to generate a human-like summary</p>')
            
            # Parameter summary
            output_html.append(f"""
            <div class="alert alert-light">
                <span><strong>Parameters:</strong> Min Length: {min_length} | Max Length: {max_length} | Sampling: {'Enabled' if use_sampling else 'Disabled'}</span>
            </div>
            """)
            
            try:
                # Load summarizer model
                summarizer = load_summarizer()
                
                if summarizer is None:
                    output_html.append("""
                    <div class="alert alert-danger">
                        <p>Failed to load the abstractive summarization model. This may be due to memory constraints or missing dependencies.</p>
                    </div>
                    """)
                else:
                    # Check length limitations
                    max_token_limit = 1024  # BART typically has 1024 token limit
                    
                    # If text is too long, warn user and truncate
                    if word_count > max_token_limit:
                        output_html.append(f"""
                        <div class="alert alert-warning">
                            <p><b>⚠️ Note:</b> Text exceeds model's length limit. Only the first ~{max_token_limit} tokens will be used for summarization.</p>
                        </div>
                        """)
                    
                    # Generate summary using the specified min_length and max_length
                    abstractive_results = summarizer(
                        text_input, 
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=use_sampling,
                        temperature=0.7 if use_sampling else 1.0,
                        top_p=0.9 if use_sampling else 1.0,
                        length_penalty=2.0
                    )
                    
                    abstractive_summary = abstractive_results[0]['summary_text']
                    
                    # Calculate reduction statistics
                    abstractive_word_count = len(abstractive_summary.split())
                    abstractive_reduction = (1 - abstractive_word_count / word_count) * 100
                    
                    # Summary Results
                    output_html.append(f"""
                    <div class="card">
                        <div class="card-header">
                            <h4 class="mb-0">Neural Summary</h4>
                        </div>
                        <div class="card-body">
                            <div style="line-height: 1.6;">
                                {abstractive_summary}
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-3">
                        <div class="col-md-4">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="text-muted">Original Length</h5>
                                    <h3 class="text-primary">{word_count} words</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="text-muted">Summary Length</h5>
                                    <h3 class="text-success">{abstractive_word_count} words</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="text-muted">Compression</h5>
                                    <h3 class="text-info">{abstractive_reduction:.1f}%</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                    """)
                    
                    # Key Terms & Topics Section
                    output_html.append('<h3 class="task-subheader">Key Topics & Terms</h3>')
                    
                    # Extract key terms with TF-IDF
                    key_terms = extract_key_terms(text_input, n=10)
                    
                    # Create layout stacked vertically: table first, then chart
                    output_html.append('<div class="row">')
                    
                    # Row 1: Key terms table (full width)
                    output_html.append('<div class="col-12">')
                    output_html.append('<h4>Key Terms</h4>')
                    
                    # Create key terms table
                    terms_df = pd.DataFrame({
                        '#': range(1, len(key_terms) + 1),
                        'Keyword': [term[0] for term in key_terms],
                        'TF-IDF Score': [f"{term[1]:.4f}" for term in key_terms]
                    })
                    
                    output_html.append(df_to_html_table(terms_df))
                    output_html.append('</div>')  # Close row 1 column
                    output_html.append('</div>')  # Close row 1
                    
                    # Row 2: Term importance chart (full width)
                    output_html.append('<div class="row mt-3">')
                    output_html.append('<div class="col-12">')
                    output_html.append('<h4>Term Importance</h4>')
                    
                    # Create horizontal bar chart of key terms
                    fig = plt.figure(figsize=(10, 8))
                    
                    # Reverse the order for bottom-to-top display
                    terms = [term[0] for term in key_terms]
                    scores = [term[1] for term in key_terms]
                    
                    # Sort by score for better visualization
                    sorted_data = sorted(zip(terms, scores), key=lambda x: x[1])
                    terms = [x[0] for x in sorted_data]
                    scores = [x[1] for x in sorted_data]
                    
                    # Create horizontal bar chart
                    plt.barh(terms, scores, color='#1976D2')
                    plt.xlabel('TF-IDF Score')
                    plt.ylabel('Keyword')
                    plt.title('Key Terms by TF-IDF Score')
                    plt.tight_layout()
                    
                    output_html.append(fig_to_html(fig))
                    
                    output_html.append('</div>')  # Close row 2 column
                    output_html.append('</div>')  # Close row 2
            
            except Exception as e:
                output_html.append(f"""
                <div class="alert alert-danger">
                    <h4>Abstractive Summarization Error</h4>
                    <p>Failed to perform abstractive summarization: {str(e)}</p>
                </div>
                """)
            
            # Extractive Summarization
            output_html.append('<h3 class="task-subheader">Extractive Summarization</h3>')
            output_html.append("""
            <div class="alert alert-light">
                <p class="mb-0">
                Extractive summarization works by identifying important sentences in the text and extracting them to form a summary.
                This implementation uses a variant of the TextRank algorithm, which is based on Google's PageRank.
                </p>
            </div>
            """)
            
            # Perform TextRank Summarization
            extractive_summary = textrank_summarize(text_input, num_sentences=min(3, max(1, len(sentences) // 3)))
            
            # Clean up the placeholder separator
            extractive_summary = extractive_summary.replace("SENTBREAKOS.OS", " ")
            
            # Calculate reduction statistics
            extractive_word_count = len(extractive_summary.split())
            extractive_reduction = (1 - extractive_word_count / word_count) * 100
            
            output_html.append(f"""
            <div class="alert alert-success">
                <h4>Extractive Summary ({extractive_reduction:.1f}% reduction)</h4>
                <div style="line-height: 1.6;">
                    {extractive_summary}
                </div>
            </div>
            """)
            
            # Sentence importance visualization
            output_html.append('<h4>Sentence Importance</h4>')
            output_html.append('<p>The graph below shows the relative importance of each sentence based on the TextRank algorithm:</p>')
            
            # Get sentence scores from TextRank
            sentence_scores = textrank_sentence_scores(text_input)
            
            # Sort sentences by their original order
            sentence_items = list(sentence_scores.items())
            sentence_items.sort(key=lambda x: int(x[0].split('_')[1]))
            
            # Create visualization
            fig = plt.figure(figsize=(10, 6))
            bars = plt.bar(
                [f"Sent {item[0].split('_')[1]}" for item in sentence_items], 
                [item[1] for item in sentence_items], 
                color='#1976D2'
            )
            
            # Highlight selected sentences
            selected_indices = [int(idx.split('_')[1]) for idx in sentence_scores.keys() if idx in extractive_summary.split('SENTBREAKOS.OS')]
            for i, bar in enumerate(bars):
                if i+1 in selected_indices:
                    bar.set_color('#4CAF50')
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           'Selected', ha='center', va='bottom', fontsize=8, rotation=90)
            
            plt.xlabel('Sentence')
            plt.ylabel('Importance Score')
            plt.title('Sentence Importance Based on TextRank')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            output_html.append(fig_to_html(fig))
            
            # Compare the two approaches
            output_html.append('<h3 class="task-subheader">Summary Comparison</h3>')
            
            # Calculate overlap between summaries
            extractive_words = set(re.findall(r'\b\w+\b', extractive_summary.lower()))
            abstractive_words = set(re.findall(r'\b\w+\b', abstractive_summary.lower()))
            common_words = extractive_words.intersection(abstractive_words)
            
            if len(extractive_words) > 0 and len(abstractive_words) > 0:
                overlap_percentage = len(common_words) / ((len(extractive_words) + len(abstractive_words)) / 2) * 100
            else:
                overlap_percentage = 0
            
            # Create comparison table
            comparison_data = {
                'Metric': ['Word Count', 'Reduction %', 'Sentences', 'Words per Sentence', 'Unique Words'],
                'Extractive': [
                    extractive_word_count,
                    f"{extractive_reduction:.1f}%",
                    len(nltk.sent_tokenize(extractive_summary)),
                    f"{extractive_word_count / max(1, len(nltk.sent_tokenize(extractive_summary))):.1f}",
                    len(extractive_words)
                ],
                'Abstractive': [
                    abstractive_word_count,
                    f"{abstractive_reduction:.1f}%",
                    len(nltk.sent_tokenize(abstractive_summary)),
                    f"{abstractive_word_count / max(1, len(nltk.sent_tokenize(abstractive_summary))):.1f}",
                    len(abstractive_words)
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            output_html.append('<div class="row">')
            
            # Column 1: Comparison table
            output_html.append('<div class="col-md-6">')
            output_html.append('<h4>Summary Statistics</h4>')
            output_html.append(df_to_html_table(comparison_df))
            output_html.append('</div>')
            
            # Column 2: Venn diagram of word overlap
            output_html.append('<div class="col-md-6">')
            output_html.append('<h4>Word Overlap Visualization</h4>')
            
            # Create Venn diagram
            fig = plt.figure(figsize=(8, 6))
            venn = venn2(
                subsets=(
                    len(extractive_words - abstractive_words),
                    len(abstractive_words - extractive_words),
                    len(common_words)
                ),
                set_labels=('Extractive', 'Abstractive')
            )
            
            # Set colors
            venn.get_patch_by_id('10').set_color('#4CAF50')
            venn.get_patch_by_id('01').set_color('#03A9F4')
            venn.get_patch_by_id('11').set_color('#9C27B0')
            
            plt.title('Word Overlap Between Summaries')
            plt.text(0, -0.25, f"Overlap: {overlap_percentage:.1f}%", ha='center')
            
            output_html.append(fig_to_html(fig))
            
            # Show key shared and unique words
            shared_words_list = list(common_words)
            extractive_only = list(extractive_words - abstractive_words)
            abstractive_only = list(abstractive_words - extractive_words)
            
            # Limit the number of words shown
            max_words = 10
            
            output_html.append(f"""
            <div class="mt-3">
                <h5>Key Shared Words ({min(max_words, len(shared_words_list))} of {len(shared_words_list)})</h5>
                <div class="d-flex flex-wrap gap-1 mb-2">
                    {' '.join([f'<span class="badge bg-primary">{word}</span>' for word in shared_words_list[:max_words]])}
                </div>
                
                <h5>Unique to Extractive ({min(max_words, len(extractive_only))} of {len(extractive_only)})</h5>
                <div class="d-flex flex-wrap gap-1 mb-2">
                    {' '.join([f'<span class="badge bg-success">{word}</span>' for word in extractive_only[:max_words]])}
                </div>
                
                <h5>Unique to Abstractive ({min(max_words, len(abstractive_only))} of {len(abstractive_only)})</h5>
                <div class="d-flex flex-wrap gap-1 mb-2">
                    {' '.join([f'<span class="badge bg-info">{word}</span>' for word in abstractive_only[:max_words]])}
                </div>
            </div>
            """)
            
            output_html.append('</div>')  # Close column 2
            output_html.append('</div>')  # Close row
    
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h3>Error</h3>
            <p>Failed to summarize text: {str(e)}</p>
        </div>
        """)
    
    # About Text Summarization section
    output_html.append("""
    <div class="card mt-4">
        <div class="card-header">
            <h4 class="mb-0">
                <i class="fas fa-info-circle"></i>
                About Text Summarization
            </h4>
        </div>
        <div class="card-body">
            <h5>What is Text Summarization?</h5>
            
            <p>Text summarization is the process of creating a shorter version of a text while preserving its key information
            and meaning. It helps users quickly grasp the main points without reading the entire document.</p>
            
            <h5>Two Main Approaches:</h5>
            
            <ul>
                <li><b>Extractive Summarization:</b> Selects and extracts existing sentences from the source text based on their importance</li>
                <li><b>Abstractive Summarization:</b> Generates new sentences that capture the meaning of the source text (similar to how humans write summaries)</li>
            </ul>
            
            <h5>Applications:</h5>
            
            <ul>
                <li><b>News digests</b> - Quick summaries of news articles</li>
                <li><b>Research papers</b> - Condensing long academic papers</li>
                <li><b>Legal documents</b> - Summarizing complex legal text</li>
                <li><b>Meeting notes</b> - Extracting key points from discussions</li>
                <li><b>Content curation</b> - Creating snippets for content recommendations</li>
            </ul>
        </div>
    </div>
    """)
    
    output_html.append('</div>')  # Close result-area div
    
    return '\n'.join(output_html)

def extract_key_terms(text, n=10):
    """Extract key terms using TF-IDF"""
    try:
        # Tokenize and preprocess
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        # Tokenize and clean text
        words = word_tokenize(text.lower())
        words = [lemmatizer.lemmatize(word) for word in words 
                if word.isalnum() and word not in stop_words and len(word) > 2]
        
        # Create document for TF-IDF
        document = [' '.join(words)]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(document)
        
        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Create term-score pairs and sort by score
        term_scores = [(term, score) for term, score in zip(feature_names, scores)]
        term_scores.sort(key=lambda x: x[1], reverse=True)
        
        return term_scores[:n]
    except Exception as e:
        print(f"Error extracting key terms: {str(e)}")
        return [("term", 0.0) for _ in range(n)]  # Return empty placeholder

# TextRank extractive summarization algorithm
def textrank_summarize(text, num_sentences=3):
    """Generate an extractive summary using TextRank algorithm"""
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # If text is too short, return the original text
    if len(sentences) <= num_sentences:
        return text
    
    # Build a graph of sentences with similarity edges
    sentence_scores = textrank_sentence_scores(text)
    
    # Sort sentences by score
    ranked_sentences = sorted([(score, i, s) for i, (s, score) in enumerate(zip(sentences, sentence_scores.values()))], reverse=True)
    
    # Select top sentences based on score
    selected_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
    
    # Combine selected sentences
    summary = "SENTBREAKOS.OS".join([s[2] for s in selected_sentences])
    
    return summary

def textrank_sentence_scores(text):
    """Generate sentence scores using TextRank algorithm"""
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Create sentence IDs
    sentence_ids = [f"sentence_{i+1}" for i in range(len(sentences))]
    
    # Create sentence graph
    G = nx.Graph()
    
    # Add nodes
    for sentence_id in sentence_ids:
        G.add_node(sentence_id)
    
    # Remove stopwords and preprocess sentences
    stop_words = set(stopwords.words('english'))
    sentence_words = []
    
    for sentence in sentences:
        words = [word.lower() for word in word_tokenize(sentence) if word.lower() not in stop_words and word.isalnum()]
        sentence_words.append(words)
    
    # Add edges based on sentence similarity
    for i in range(len(sentence_ids)):
        for j in range(i+1, len(sentence_ids)):
            similarity = sentence_similarity(sentence_words[i], sentence_words[j])
            if similarity > 0:
                G.add_edge(sentence_ids[i], sentence_ids[j], weight=similarity)
    
    # Run PageRank
    scores = nx.pagerank(G)
    
    return scores

def sentence_similarity(words1, words2):
    """Calculate similarity between two sentences based on word overlap"""
    if not words1 or not words2:
        return 0
    
    # Convert to sets for intersection
    set1 = set(words1)
    set2 = set(words2)
    
    # Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0
    return intersection / union
