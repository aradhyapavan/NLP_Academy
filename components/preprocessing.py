import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
import string
import base64
import io
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud
from utils.model_loader import download_nltk_resources
from utils.helpers import fig_to_html, df_to_html_table
from nltk.util import ngrams

def preprocessing_handler(text_input):
    """Generate HTML for text preprocessing display"""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Text Preprocessing</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-info-circle"></i>
    Text preprocessing is the process of cleaning and transforming raw text into a format that can be easily analyzed by NLP models.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Tools & Libraries Used:</h4>
        <ul>
            <li><b>NLTK</b> - For stopwords, tokenization, stemming and lemmatization</li>
            <li><b>Regular Expressions</b> - For pattern matching and text cleaning</li>
            <li><b>WordCloud</b> - For visualizing word frequency</li>
        </ul>
    </div>
    """)
    
    # Ensure NLTK resources are downloaded
    download_nltk_resources()
    
    try:
        # Original Text
        output_html.append('<h3 class="task-subheader">Original Text</h3>')
        output_html.append(f'<div class="card"><div class="card-body"><div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 500px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">{text_input}</div></div></div>')
        
        # Text statistics
        word_count = len(text_input.split())
        char_count = len(text_input)
        sentence_count = len(nltk.sent_tokenize(text_input))
        
        stats_html = f"""
        <div class="stats-container">
            <div class="row">
                <div class="col-md-4">
                    <div class="card text-center stats-card">
                        <div class="card-body">
                            <h3 class="metric-blue">{word_count}</h3>
                            <p>Words</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card text-center stats-card">
                        <div class="card-body">
                            <h3 class="metric-green">{char_count}</h3>
                            <p>Characters</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card text-center stats-card">
                        <div class="card-body">
                            <h3 class="metric-orange">{sentence_count}</h3>
                            <p>Sentences</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        output_html.append(stats_html)
        
        # NEW SECTION: Text Cleaning with Regular Expressions
        output_html.append('<div class="section-divider"></div>')
        output_html.append('<h3 class="task-subheader">Text Cleaning with Regular Expressions</h3>')
        
        output_html.append("""
        <div class="alert alert-light">
            <p>Regular expressions (regex) provide powerful pattern matching capabilities for cleaning and processing text data. 
            Common text cleaning tasks include removing URLs, HTML tags, special characters, and normalizing text formats.</p>
        </div>
        """)
        
        # Several regex cleaning examples
        url_pattern = r'https?://\S+|www\.\S+'
        html_pattern = r'<.*?>'
        whitespace_pattern = r'\s+'
        email_pattern = r'\S+@\S+'
        
        # Original text for comparison
        text_cleaned = text_input
        
        # 1. Remove URLs
        urls_cleaned = re.sub(url_pattern, '[URL]', text_cleaned)
        
        # 2. Remove HTML tags
        html_cleaned = re.sub(html_pattern, '', urls_cleaned)
        
        # 3. Remove extra whitespace
        whitespace_cleaned = re.sub(whitespace_pattern, ' ', html_cleaned).strip()
        
        # 4. Remove email addresses
        email_cleaned = re.sub(email_pattern, '[EMAIL]', whitespace_cleaned)
        
        # 5. Fix common contractions
        contractions = {
            r"won't": "will not",
            r"can't": "cannot",
            r"n't": " not",
            r"'re": " are",
            r"'s": " is",
            r"'d": " would",
            r"'ll": " will",
            r"'t": " not",
            r"'ve": " have",
            r"'m": " am"
        }
        
        contraction_cleaned = email_cleaned
        for pattern, replacement in contractions.items():
            contraction_cleaned = re.sub(pattern, replacement, contraction_cleaned)
        
        # Display the regex cleaning examples in a table
        output_html.append("""
        <h4>Regex Text Cleaning Operations</h4>
        <div class="table-responsive">
        <table class="table table-striped">
            <thead class="table-primary">
                <tr>
                    <th>Operation</th>
                    <th>Regex Pattern</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>URL Removal</td>
                    <td><code>https?://\\S+|www\\.\\S+</code></td>
                    <td>Removes or replaces web URLs in text</td>
                </tr>
                <tr>
                    <td>HTML Tag Removal</td>
                    <td><code>&lt;.*?&gt;</code></td>
                    <td>Strips HTML/XML markup tags</td>
                </tr>
                <tr>
                    <td>Whitespace Normalization</td>
                    <td><code>\\s+</code></td>
                    <td>Replaces multiple spaces, tabs, and newlines with a single space</td>
                </tr>
                <tr>
                    <td>Email Anonymization</td>
                    <td><code>\\S+@\\S+</code></td>
                    <td>Redacts email addresses for privacy</td>
                </tr>
                <tr>
                    <td>Contraction Expansion</td>
                    <td><code>Multiple patterns</code></td>
                    <td>Expands contractions like "don't" to "do not"</td>
                </tr>
            </tbody>
        </table>
        </div>
        """)
        
        # Example of cleaned text
        output_html.append("""
        <h4>Example of Text After Regex Cleaning</h4>
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Before Cleaning</h5>
                    </div>
                    <div class="card-body">
                        <div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">""")
        output_html.append(f"{text_input}")
        output_html.append("""</div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">After Regex Cleaning</h5>
                    </div>
                    <div class="card-body">
                        <div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">""")
        output_html.append(f"{contraction_cleaned}")
        output_html.append("""</div>
                    </div>
                </div>
            </div>
        </div>
        """)
        
        output_html.append("""
        <div class="alert alert-success">
            <h4><i class="fas fa-lightbulb"></i> Why Use Regex for Text Cleaning?</h4>
            <ul>
                <li><b>Precision:</b> Regular expressions allow for precise pattern matching</li>
                <li><b>Flexibility:</b> Can be customized for domain-specific cleaning needs</li>
                <li><b>Efficiency:</b> Processes text in a single pass for better performance</li>
                <li><b>Standardization:</b> Creates consistent formatting across documents</li>
            </ul>
        </div>
        """)
        
        # Word length distribution
        word_lengths = [len(word) for word in text_input.split()]
        fig = plt.figure(figsize=(10, 4))
        plt.hist(word_lengths, bins=range(1, max(word_lengths) + 2), alpha=0.7, color='#1976D2')
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
        plt.title('Word Length Distribution')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_html.append('<div class="section-divider"></div>')
        output_html.append('<h3 class="task-subheader">Word Length Distribution</h3>')
        output_html.append(fig_to_html(fig))
        
        # Case Normalization
        output_html.append('<div class="section-divider"></div>')
        output_html.append('<h3 class="task-subheader">Case Normalization</h3>')
        
        lowercase_text = text_input.lower()
        uppercase_text = text_input.upper()
        
        case_html = f"""
        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Original Text</h5>
                    </div>
                    <div class="card-body">
                        <div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">{text_input}</div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Lowercase Text</h5>
                    </div>
                    <div class="card-body">
                        <div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">{lowercase_text}</div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Uppercase Text</h5>
                    </div>
                    <div class="card-body">
                        <div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">{uppercase_text}</div>
                    </div>
                </div>
            </div>
        </div>
        """
        output_html.append(case_html)
        
        # Remove Punctuation & Special Characters
        output_html.append('<div class="section-divider"></div>')
        output_html.append('<h3 class="task-subheader">Punctuation & Special Characters Removal</h3>')
        
        # Count original punctuation
        punc_count = sum([1 for char in text_input if char in string.punctuation])
        
        # Remove punctuation
        no_punct_text = re.sub(r'[^\w\s]', '', text_input)
        
        punct_html = f"""
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Original Text</h5>
                    </div>
                    <div class="card-body">
                        <div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">{text_input}</div>
                        <small class="text-muted">Contains {punc_count} punctuation marks</small>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Without Punctuation</h5>
                    </div>
                    <div class="card-body">
                        <div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">{no_punct_text}</div>
                        <small class="text-muted">Removed {punc_count} punctuation marks</small>
                    </div>
                </div>
            </div>
        </div>
        """
        output_html.append(punct_html)
        
        # Show removed punctuation
        punct_chars = [char for char in text_input if char in string.punctuation]
        punct_freq = Counter(punct_chars)
        
        if punct_freq:
            output_html.append('<h4>Punctuation Distribution</h4>')
            
            fig = plt.figure(figsize=(10, 4))
            plt.bar(punct_freq.keys(), punct_freq.values(), color='#1976D2')
            plt.xlabel('Punctuation')
            plt.ylabel('Frequency')
            plt.title('Punctuation Distribution')
            plt.tight_layout()
            
            output_html.append(fig_to_html(fig))
        
        # Tokenization
        output_html.append('<div class="section-divider"></div>')
        output_html.append('<h3 class="task-subheader">Tokenization</h3>')
        
        # Word tokenization
        words = nltk.word_tokenize(text_input)
        
        # Create a multi-column layout for word tokens
        output_html.append('<h4>Word Tokens</h4>')
        output_html.append(f'<p>Total tokens: {len(words)} (showing first 50)</p>')
        
        # Create a multi-column table layout
        tokens_html = """
        <div class="table-responsive">
        <table class="table table-striped table-hover" style="table-layout: fixed;">
            <thead class="table-primary">
                <tr>
                    <th style="width: 8%;">#</th>
                    <th style="width: 25%;">Token</th>
                    <th style="width: 12%;">Length</th>
                    <th style="width: 8%;">#</th>
                    <th style="width: 25%;">Token</th>
                    <th style="width: 12%;">Length</th>
                    <th style="width: 8%;">#</th>
                    <th style="width: 25%;">Token</th>
                    <th style="width: 12%;">Length</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Create rows with 3 tokens per row
        for i in range(0, min(50, len(words)), 3):
            tokens_html += "<tr>"
            for j in range(3):
                if i + j < min(50, len(words)):
                    token = words[i + j]
                    tokens_html += f'<td>{i + j + 1}</td><td><code>{token}</code></td><td><span class="badge bg-secondary">{len(token)}</span></td>'
                else:
                    tokens_html += '<td></td><td></td><td></td>'
            tokens_html += "</tr>"
        
        tokens_html += """
            </tbody>
        </table>
        </div>
        """
        
        output_html.append(tokens_html)
        
        # Sentence tokenization
        sentences = nltk.sent_tokenize(text_input)
        
        output_html.append('<h4>Sentence Tokens</h4>')
        output_html.append(f'<p>Total sentences: {len(sentences)}</p>')
        
        for i, sentence in enumerate(sentences[:5]):
            output_html.append(f'<div class="card mb-2"><div class="card-body"><strong>{i+1}.</strong> {sentence}</div></div>')
        
        if len(sentences) > 5:
            output_html.append(f'<p class="text-muted">... and {len(sentences) - 5} more sentences.</p>')
        
        # Stopwords Removal
        output_html.append('<div class="section-divider"></div>')
        output_html.append('<h3 class="task-subheader">Stopwords Removal</h3>')
        
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Count stopwords
        stopword_count = len(words) - len(filtered_words)
        stopword_percentage = (stopword_count / len(words)) * 100 if words else 0
        
        output_html.append(f"""
        <div class="row mb-3">
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <h5>Original Words</h5>
                        <h3 class="text-primary">{len(words)}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <h5>After Stopword Removal</h5>
                        <h3 class="text-success">{len(filtered_words)}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <h5>Stopwords Removed</h5>
                        <h3 class="text-warning">{stopword_count} ({stopword_percentage:.1f}%)</h3>
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # Display common stopwords in the text
        text_stopwords = [word for word in words if word.lower() in stop_words]
        stop_freq = Counter(text_stopwords).most_common(10)
        
        if stop_freq:
            output_html.append('<h4>Most Common Stopwords in Text</h4>')
            
            # Create a multi-column layout for stopwords
            stopwords_html = """
            <div class="table-responsive">
            <table class="table table-striped table-hover" style="table-layout: fixed;">
                <thead class="table-primary">
                    <tr>
                        <th style="width: 10%;">#</th>
                        <th style="width: 35%;">Stopword</th>
                        <th style="width: 15%;">Frequency</th>
                        <th style="width: 10%;">#</th>
                        <th style="width: 35%;">Stopword</th>
                        <th style="width: 15%;">Frequency</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            # Create rows with 2 stopwords per row
            for i in range(0, len(stop_freq), 2):
                stopwords_html += "<tr>"
                for j in range(2):
                    if i + j < len(stop_freq):
                        stopword, freq = stop_freq[i + j]
                        stopwords_html += f'<td>{i + j + 1}</td><td><code>{stopword}</code></td><td><span class="badge bg-warning">{freq}</span></td>'
                    else:
                        stopwords_html += '<td></td><td></td><td></td>'
                stopwords_html += "</tr>"
            
            stopwords_html += """
                </tbody>
            </table>
            </div>
            """
            
            output_html.append(stopwords_html)
            
            # Visualization of before and after
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Before
            ax[0].hist([len(word) for word in words], bins=range(1, 15), alpha=0.7, color='#1976D2')
            ax[0].set_title('Word Length Before Stopword Removal')
            ax[0].set_xlabel('Word Length')
            ax[0].set_ylabel('Frequency')
            
            # After
            ax[1].hist([len(word) for word in filtered_words], bins=range(1, 15), alpha=0.7, color='#4CAF50')
            ax[1].set_title('Word Length After Stopword Removal')
            ax[1].set_xlabel('Word Length')
            ax[1].set_ylabel('Frequency')
            
            plt.tight_layout()
            output_html.append(fig_to_html(fig))
        
        # Stemming and Lemmatization
        output_html.append('<div class="section-divider"></div>')
        output_html.append('<h3 class="task-subheader">Stemming & Lemmatization</h3>')
        
        # Apply stemming (Porter Stemmer)
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in filtered_words[:100]]  # Limit to first 100 words for performance
        
        # Apply lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words[:100]]  # Limit to first 100 words
        
        # Create comparison DataFrame
        comparison_data = []
        for i in range(min(20, len(filtered_words))):  # Show first 20 examples
            if i < len(filtered_words) and filtered_words[i].isalpha():  # Only include alphabetic words
                comparison_data.append({
                    'Original': filtered_words[i],
                    'Stemmed': stemmer.stem(filtered_words[i]),
                    'Lemmatized': lemmatizer.lemmatize(filtered_words[i])
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        output_html.append('<h4>Stemming vs. Lemmatization Comparison</h4>')
        
        # Create a custom table for stemming vs lemmatization comparison
        comparison_html = """
        <div class="table-responsive">
        <table class="table table-striped table-hover" style="table-layout: fixed;">
            <thead class="table-primary">
                <tr>
                    <th style="width: 30%;">Original</th>
                    <th style="width: 35%;">Stemmed</th>
                    <th style="width: 35%;">Lemmatized</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add comparison data rows
        for _, row in comparison_df.iterrows():
            comparison_html += f"""
            <tr>
                <td><code>{row['Original']}</code></td>
                <td><code>{row['Stemmed']}</code></td>
                <td><code>{row['Lemmatized']}</code></td>
            </tr>
            """
        
        comparison_html += """
            </tbody>
        </table>
        </div>
        """
        
        output_html.append(comparison_html)
        
        output_html.append("""
        <div class="alert alert-success">
            <h4><i class="fas fa-lightbulb"></i> Stemming vs. Lemmatization</h4>
            <ul>
                <li><b>Stemming</b> - Cuts off word endings based on common patterns, faster but less accurate</li>
                <li><b>Lemmatization</b> - Uses vocabulary and morphological analysis, slower but produces actual words</li>
            </ul>
        </div>
        """)

        # NEW SECTION: N-gram Analysis
        output_html.append('<div class="section-divider"></div>')
        output_html.append('<h3 class="task-subheader">N-gram Analysis</h3>')
        
        output_html.append("""
        <div class="alert alert-light">
            <p>N-grams are contiguous sequences of n items from text. In NLP, they are used to capture word patterns and relationships,
            and are helpful for language modeling, prediction, and feature extraction.</p>
        </div>
        """)
        
        # Process text for n-grams (use filtered_words to avoid stopwords)
        # Convert to lowercase for consistency
        clean_words = [word.lower() for word in filtered_words if word.isalnum()]
        
        # Generate n-grams
        bigrams_list = list(ngrams(clean_words, 2))
        trigrams_list = list(ngrams(clean_words, 3))
        
        # Count frequencies
        bigram_freq = Counter(bigrams_list)
        trigram_freq = Counter(trigrams_list)
        
        # Get most common
        common_bigrams = bigram_freq.most_common(15)
        common_trigrams = trigram_freq.most_common(15)
        
        # Format for display
        bigram_labels = [' '.join(bg) for bg, _ in common_bigrams]
        bigram_values = [count for _, count in common_bigrams]
        
        trigram_labels = [' '.join(tg) for tg, _ in common_trigrams]
        trigram_values = [count for _, count in common_trigrams]
        
        # Create DataFrames for display
        bigram_df = pd.DataFrame({
            'Bigram': [' '.join(bg) for bg, _ in common_bigrams], 
            'Frequency': [count for _, count in common_bigrams]
        })
        
        trigram_df = pd.DataFrame({
            'Trigram': [' '.join(tg) for tg, _ in common_trigrams], 
            'Frequency': [count for _, count in common_trigrams]
        })
        
        # Explanation of n-grams
        output_html.append("""
        <div class="alert alert-info">
            <h4>What are N-grams?</h4>
            <ul>
                <li><b>Unigrams</b> - Single words (e.g., "climate")</li>
                <li><b>Bigrams</b> - Two consecutive words (e.g., "climate change")</li>
                <li><b>Trigrams</b> - Three consecutive words (e.g., "global climate change")</li>
            </ul>
            <p>N-grams capture contextual relationships between words and are valuable for many NLP tasks including language modeling, 
            machine translation, speech recognition, and text classification.</p>
        </div>
        """)
        
        # Create visualizations for bigrams and trigrams
        if bigram_labels and len(bigram_values) > 0:
            # Bigram visualization
            output_html.append('<h4>Most Common Bigrams</h4>')
            
            fig = plt.figure(figsize=(10, 6))
            plt.barh(range(len(bigram_labels)), bigram_values, align='center', color='#1976D2')
            plt.yticks(range(len(bigram_labels)), bigram_labels)
            plt.xlabel('Frequency')
            plt.title('Most Common Bigrams')
            plt.tight_layout()
            
            output_html.append(fig_to_html(fig))
            
            # Create a multi-column layout for bigrams
            bigram_html = """
            <div class="table-responsive">
            <table class="table table-striped table-hover" style="table-layout: fixed;">
                <thead class="table-primary">
                    <tr>
                        <th style="width: 10%;">#</th>
                        <th style="width: 35%;">Bigram</th>
                        <th style="width: 15%;">Freq</th>
                        <th style="width: 10%;">#</th>
                        <th style="width: 35%;">Bigram</th>
                        <th style="width: 15%;">Freq</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            # Create rows with 2 bigrams per row
            for i in range(0, len(common_bigrams), 2):
                bigram_html += "<tr>"
                for j in range(2):
                    if i + j < len(common_bigrams):
                        bigram, freq = common_bigrams[i + j]
                        bigram_text = ' '.join(bigram)
                        bigram_html += f'<td>{i + j + 1}</td><td><code>{bigram_text}</code></td><td><span class="badge bg-info">{freq}</span></td>'
                    else:
                        bigram_html += '<td></td><td></td><td></td>'
                bigram_html += "</tr>"
            
            bigram_html += """
                </tbody>
            </table>
            </div>
            """
            
            output_html.append(bigram_html)
        else:
            output_html.append('<p class="text-muted">Not enough text to generate meaningful bigrams.</p>')
        
        if trigram_labels and len(trigram_values) > 0:
            # Trigram visualization
            output_html.append('<h4>Most Common Trigrams</h4>')
            
            fig = plt.figure(figsize=(10, 6))
            plt.barh(range(len(trigram_labels)), trigram_values, align='center', color='#4CAF50')
            plt.yticks(range(len(trigram_labels)), trigram_labels)
            plt.xlabel('Frequency')
            plt.title('Most Common Trigrams')
            plt.tight_layout()
            
            output_html.append(fig_to_html(fig))
            
            # Create a multi-column layout for trigrams
            trigram_html = """
            <div class="table-responsive">
            <table class="table table-striped table-hover" style="table-layout: fixed;">
                <thead class="table-primary">
                    <tr>
                        <th style="width: 10%;">#</th>
                        <th style="width: 35%;">Trigram</th>
                        <th style="width: 15%;">Freq</th>
                        <th style="width: 10%;">#</th>
                        <th style="width: 35%;">Trigram</th>
                        <th style="width: 15%;">Freq</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            # Create rows with 2 trigrams per row
            for i in range(0, len(common_trigrams), 2):
                trigram_html += "<tr>"
                for j in range(2):
                    if i + j < len(common_trigrams):
                        trigram, freq = common_trigrams[i + j]
                        trigram_text = ' '.join(trigram)
                        trigram_html += f'<td>{i + j + 1}</td><td><code>{trigram_text}</code></td><td><span class="badge bg-success">{freq}</span></td>'
                    else:
                        trigram_html += '<td></td><td></td><td></td>'
                trigram_html += "</tr>"
            
            trigram_html += """
                </tbody>
            </table>
            </div>
            """
            
            output_html.append(trigram_html)
        else:
            output_html.append('<p class="text-muted">Not enough text to generate meaningful trigrams.</p>')
            
        # Applications of N-grams
        output_html.append("""
        <div class="alert alert-info">
            <h4><i class="fas fa-lightbulb"></i> Applications of N-gram Analysis</h4>
            <ul>
                <li><b>Language Modeling</b> - Predicting the next word in a sequence</li>
                <li><b>Machine Translation</b> - Improving translation quality</li>
                <li><b>Text Classification</b> - Using n-grams as features</li>
                <li><b>Spelling Correction</b> - Suggesting correct spellings</li>
                <li><b>Information Retrieval</b> - Enhancing search results</li>
                <li><b>Sentiment Analysis</b> - Capturing phrase-level sentiments</li>
            </ul>
        </div>
        """)
        
        # Word Cloud
        output_html.append('<div class="section-divider"></div>')
        output_html.append('<h3 class="task-subheader">Word Cloud</h3>')
        
        try:
            # Create word cloud from filtered words
            wordcloud_text = ' '.join(filtered_words)
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white', 
                colormap='viridis',
                max_words=100, 
                contour_width=1, 
                contour_color='#1976D2'
            ).generate(wordcloud_text)
            
            # Display word cloud
            fig = plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            
            output_html.append(fig_to_html(fig))
            
        except Exception as e:
            output_html.append(f"<div class='alert alert-warning'>Failed to generate word cloud: {str(e)}</div>")
        
        # Word Frequency
        output_html.append('<div class="section-divider"></div>')
        output_html.append('<h3 class="task-subheader">Word Frequency Analysis</h3>')
        
        # Calculate word frequencies
        word_freq = Counter(filtered_words)
        most_common = word_freq.most_common(20)
        
        # Create DataFrame
        freq_df = pd.DataFrame(most_common, columns=['Word', 'Frequency'])
        
        # Create horizontal bar chart
        fig = plt.figure(figsize=(12, 16))
        plt.barh(range(len(most_common)), [val[1] for val in most_common], align='center', color='#1976D2')
        plt.yticks(range(len(most_common)), [val[0] for val in most_common])
        plt.xlabel('Frequency')
        plt.title('Top 20 Words')
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
        plt.tight_layout(pad=3.0)
        
        # Render chart
        output_html.append('<section class="wf-chart-section">')
        output_html.append('<div class="chart-container">')
        output_html.append(fig_to_html(fig))
        output_html.append('</div>')
        output_html.append('</section>')
        
        # Create a multi-column layout for word frequency
        freq_html = """
        <section class="wf-table-container">
        <div class="table-responsive">
        <table class="table table-striped table-hover" style="table-layout: fixed;">
            <thead class="table-primary">
                <tr>
                    <th style="width: 10%;">#</th>
                    <th style="width: 35%;">Word</th>
                    <th style="width: 15%;">Freq</th>
                    <th style="width: 10%;">#</th>
                    <th style="width: 35%;">Word</th>
                    <th style="width: 15%;">Freq</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Create rows with 2 words per row
        for i in range(0, len(most_common), 2):
            freq_html += "<tr>"
            for j in range(2):
                if i + j < len(most_common):
                    word, freq = most_common[i + j]
                    freq_html += f'<td>{i + j + 1}</td><td><code>{word}</code></td><td><span class="badge bg-primary">{freq}</span></td>'
                else:
                    freq_html += '<td></td><td></td><td></td>'
            freq_html += "</tr>"
        
        freq_html += """
            </tbody>
        </table>
        </div>
        </section>
        """
        
        output_html.append(freq_html)
    
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h3>Error</h3>
            <p>Failed to process text: {str(e)}</p>
        </div>
        """)
    
    # About text preprocessing
    output_html.append("""
    <div class="card mt-4">
        <div class="card-header">
            <h4 class="mb-0">
                <i class="fas fa-info-circle"></i>
                About Text Preprocessing
            </h4>
        </div>
        <div class="card-body">
            <h5>What is Text Preprocessing?</h5>
            
            <p>Text preprocessing is the first step in NLP pipelines that transforms raw text into a clean, structured format
            suitable for analysis. It includes various techniques to standardize text and reduce noise.</p>
            
            <h5>Common Preprocessing Steps:</h5>
            
            <ul>
                <li><b>Tokenization</b> - Splitting text into individual words or sentences</li>
                <li><b>Normalization</b> - Converting text to lowercase, removing accents, etc.</li>
                <li><b>Noise Removal</b> - Removing punctuation, special characters, HTML tags, etc.</li>
                <li><b>Stopword Removal</b> - Filtering out common words that add little meaning</li>
                <li><b>Stemming/Lemmatization</b> - Reducing words to their root forms</li>
                <li><b>Spelling Correction</b> - Fixing typos and errors</li>
            </ul>
            
            <h5>Why Preprocess Text?</h5>
            
            <ul>
                <li>Reduces dimensionality and noise in the data</li>
                <li>Standardizes text for consistent analysis</li>
                <li>Improves performance of downstream NLP tasks</li>
                <li>Makes text more suitable for machine learning models</li>
            </ul>
        </div>
    </div>
    """)
    
    output_html.append('</div>')  # Close result-area div
    
    return '\n'.join(output_html)
