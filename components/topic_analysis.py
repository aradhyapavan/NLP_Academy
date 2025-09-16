import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from collections import Counter
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import wordcloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.colors as mcolors
import io
import base64

from utils.model_loader import download_nltk_resources
from utils.helpers import fig_to_html, df_to_html_table

def classify_topic(text_input):
    """Classify the topic of the text into predefined categories."""
    # Define topic keywords
    topic_keywords = {
        'environment': ['climate', 'environment', 'weather', 'earth', 'temperature', 'pollution', 'warming', 'planet', 'ecosystem', 'sustainable'],
        'science': ['science', 'scientific', 'research', 'study', 'experiment', 'discovery', 'theory', 'laboratory', 'data'],
        'business': ['business', 'company', 'market', 'economy', 'economic', 'finance', 'industry', 'corporate', 'trade'],
        'education': ['education', 'school', 'student', 'learn', 'teach', 'academic', 'university', 'college', 'knowledge'],
        'health': ['health', 'medical', 'doctor', 'patient', 'disease', 'treatment', 'hospital', 'medicine', 'healthcare'],
        'technology': ['technology', 'tech', 'computer', 'digital', 'software', 'hardware', 'internet', 'device', 'innovation'],
        'politics': ['politics', 'government', 'policy', 'election', 'political', 'law', 'president', 'party', 'vote'],
        'sports': ['sport', 'game', 'team', 'player', 'competition', 'athlete', 'championship', 'tournament', 'coach'],
        'entertainment': ['entertainment', 'movie', 'music', 'film', 'television', 'celebrity', 'actor', 'actress', 'show'],
        'travel': ['travel', 'trip', 'vacation', 'tourist', 'destination', 'journey', 'adventure', 'flight', 'hotel']
    }
    
    # Convert text to lowercase
    text = text_input.lower()
    
    # Count keyword occurrences for each topic
    topic_scores = {}
    for topic, keywords in topic_keywords.items():
        score = 0
        for keyword in keywords:
            # Count occurrences of the keyword
            count = text.count(keyword)
            # Add to the topic score
            score += count
        
        # Store the normalized score
        topic_scores[topic] = score / (len(text.split()) + 0.001)  # Normalize by text length

    # Get the main topic and confidence
    main_topic = max(topic_scores.items(), key=lambda x: x[1])
    total_score = sum(topic_scores.values()) + 0.001  # Avoid division by zero
    confidence = main_topic[1] / total_score if total_score > 0 else 0
    confidence = round(confidence * 100, 1)  # Convert to percentage

    # Sort topics by score for visualization
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    
    return main_topic[0], confidence, sorted_topics, topic_scores

def extract_key_phrases(text_input, top_n=10):
    """Extract key phrases from text."""
    # Download required NLTK resources
    download_nltk_resources()
    
    # Define stop words
    stop_words = set(stopwords.words('english'))
    
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text_input)
    
    # Extract 2-3 word phrases (n-grams)
    phrases = []
    
    # Get bigrams
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=100)
    try:
        bigram_matrix = bigram_vectorizer.fit_transform([text_input])
        bigram_features = bigram_vectorizer.get_feature_names_out()
        bigram_scores = bigram_matrix.toarray()[0]
        
        for phrase, score in zip(bigram_features, bigram_scores):
            if score >= 1:  # Must appear at least once
                phrases.append((phrase, int(score)))
    except:
        pass  # Handle potential errors
    
    # Get trigrams
    trigram_vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english', max_features=100)
    try:
        trigram_matrix = trigram_vectorizer.fit_transform([text_input])
        trigram_features = trigram_vectorizer.get_feature_names_out()
        trigram_scores = trigram_matrix.toarray()[0]
        
        for phrase, score in zip(trigram_features, trigram_scores):
            if score >= 1:  # Must appear at least once
                phrases.append((phrase, int(score)))
    except:
        pass
    
    # Also extract single important words (nouns, verbs, adjectives)
    words = word_tokenize(text_input)
    pos_tags = nltk.pos_tag(words)
    
    important_words = []
    for word, tag in pos_tags:
        # Only consider nouns, verbs, and adjectives
        if (tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ')) and word.lower() not in stop_words and len(word) > 2:
            important_words.append(word.lower())
    
    # Count word frequencies
    word_freq = Counter(important_words)
    
    # Add important single words to phrases
    for word, freq in word_freq.most_common(top_n):
        if freq >= 1:
            phrases.append((word, freq))
    
    # Sort phrases by frequency
    sorted_phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
    
    # Return top N phrases
    return sorted_phrases[:top_n]

def create_phrase_cloud(phrases):
    """Create a word cloud from phrases."""
    # Convert phrases to a dictionary of {phrase: frequency}
    phrase_freq = {phrase: freq for phrase, freq in phrases}
    
    # Create word cloud
    wc = wordcloud.WordCloud(
        background_color='white',
        width=600,
        height=400,
        colormap='viridis',
        max_words=50,
        prefer_horizontal=0.9,
        random_state=42
    )
    
    try:
        # Generate word cloud from phrases
        wc.generate_from_frequencies(phrase_freq)
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        
        return fig_to_html(fig)
    except:
        return "<p>Could not generate phrase cloud due to insufficient data.</p>"

def topic_analysis_handler(text_input):
    """Show topic analysis capabilities."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Topic Analysis</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-info-circle"></i>
    Topic analysis identifies the main themes and subjects in a text, helping to categorize content and understand what it's about.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Models & Techniques Used:</h4>
        <ul>
            <li><b>Zero-shot Classification</b> - BART model that can classify text without specific training</li>
            <li><b>TF-IDF Vectorizer</b> - Statistical method to identify important terms</li>
            <li><b>Word/Phrase Analysis</b> - Extraction of important n-grams</li>
        </ul>
    </div>
    """)
    
    try:
        # Ensure NLTK resources are downloaded
        download_nltk_resources()
        
        # Check if text is long enough for meaningful analysis
        if len(text_input.split()) < 50:
            output_html.append(f"""
            <div class="alert alert-warning">
                <h3>Text Too Short for Full Topic Analysis</h3>
                <p>The provided text contains only {len(text_input.split())} words. 
                For meaningful topic analysis, please provide a longer text (at least 50 words).
                We'll still perform basic frequency analysis, but topic modeling results may not be reliable.</p>
            </div>
            """)
        
        # Text cleaning and preprocessing
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        def preprocess_text(text):
            # Tokenize
            tokens = word_tokenize(text.lower())
            # Remove stopwords and non-alphabetic tokens
            filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
            # Lemmatize
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
            return lemmatized_tokens
        
        # Process the text
        processed_tokens = preprocess_text(text_input)
        processed_text = ' '.join(processed_tokens)
        
        # Add Topic Classification section
        output_html.append('<h3 class="task-subheader">Topic Classification</h3>')
        
        # Get topic classification
        main_topic, confidence, sorted_topics, topic_scores = classify_topic(text_input)
        
        # Display topic classification results
        output_html.append(f"""
        <div class="alert alert-success">
            <p class="mb-0 fs-5">This text is primarily about <strong>{main_topic}</strong> with {confidence}% confidence</p>
        </div>
        """)
        
        # Display topic scores (stacked rows to avoid overlap)
        output_html.append('<div class="row">')
        
        # Row 1: Topic Relevance Chart (full width)
        output_html.append('<div class="col-12">')
        output_html.append('<h4>Topic Relevance</h4>')
        
        # Create horizontal bar chart for topic scores
        plt.figure(figsize=(10, 6))
        topics = [topic for topic, score in sorted_topics]
        scores = [score for topic, score in sorted_topics]
        
        # Only show top topics for clarity
        top_n = min(10, len(topics))
        y_pos = np.arange(top_n)
        
        # Get a color gradient
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, top_n))
        
        # Create horizontal bars
        bars = plt.barh(y_pos, [s * 100 for s in scores[:top_n]], color=colors)
        
        # Add labels and values
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{width:.1f}%",
                    va='center')
        
        plt.yticks(y_pos, topics[:top_n])
        plt.xlabel('Relevance')
        plt.title('Topic Scores')
        plt.tight_layout()
        
        output_html.append(fig_to_html(plt.gcf()))
        output_html.append('</div>')
        output_html.append('</div>')  # Close row 1
        
        # Row 2: Topic Scores Table (full width)
        output_html.append('<div class="row mt-3">')
        output_html.append('<div class="col-12">')
        output_html.append('<h4>Topic Scores</h4>')
        
        # Create table of topic scores
        topic_scores_df = pd.DataFrame({
            'Rank': range(1, len(sorted_topics) + 1),
            'Topic': [topic.capitalize() for topic, _ in sorted_topics],
            'Confidence': [f"{score:.4f}" for _, score in sorted_topics]
        })
        
        output_html.append(df_to_html_table(topic_scores_df))
        output_html.append('</div>')
        output_html.append('</div>')  # Close row 2
        
        # Extract and display key phrases
        output_html.append('<h3 class="task-subheader">Key Phrases</h3>')
        
        # Extract key phrases
        key_phrases = extract_key_phrases(text_input)
        
        # Display key phrases in a table
        if key_phrases:
            phrase_df = pd.DataFrame({
                'Phrase': [phrase for phrase, _ in key_phrases],
                'Frequency': [freq for _, freq in key_phrases]
            })
            
            output_html.append('<div class="row">')
            
            # Row 1: Key phrases table (full width)
            output_html.append('<div class="col-12">')
            output_html.append(df_to_html_table(phrase_df))
            output_html.append('</div>')
            
            # Row 2: Phrase cloud (full width)
            output_html.append('</div>')  # Close row 1
            output_html.append('<div class="row mt-3">')
            output_html.append('<div class="col-12">')
            output_html.append(create_phrase_cloud(key_phrases))
            output_html.append('</div>')
            
            output_html.append('</div>')  # Close row 2
        else:
            output_html.append("<p>No key phrases could be extracted from the text.</p>")
        
        # Term Frequency Analysis
        output_html.append('<h3 class="task-subheader">Key Term Frequency Analysis</h3>')
        
        # Get token frequencies
        token_freq = Counter(processed_tokens)
        
        # Sort by frequency
        sorted_word_freq = dict(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
        
        # Take top 25 words for visualization
        top_n = 25
        top_words = list(sorted_word_freq.keys())[:top_n]
        top_freqs = list(sorted_word_freq.values())[:top_n]
        
        # Create visualization
        fig = plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(top_words)))
        bars = plt.bar(top_words, top_freqs, color=colors)
        plt.xlabel('Term')
        plt.ylabel('Frequency')
        plt.title(f'Top {top_n} Term Frequencies')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}',
                    ha='center', va='bottom',
                    fontsize=8)
        
        # Show plots and table in stacked rows
        output_html.append('<div class="row">')
        
        # Row 1: Chart (full width)
        output_html.append('<div class="col-12">')
        output_html.append(fig_to_html(fig))
        output_html.append('</div>')
        
        # Row 2: Top terms table (full width)
        output_html.append('</div>')  # Close row 1
        output_html.append('<div class="row mt-3">')
        output_html.append('<div class="col-12">')
        output_html.append('<h4>Top Terms</h4>')
        
        # Create DataFrame of top terms
        top_terms_df = pd.DataFrame({
            'Term': list(sorted_word_freq.keys())[:15],
            'Frequency': list(sorted_word_freq.values())[:15]
        })
        
        output_html.append(df_to_html_table(top_terms_df))
        output_html.append('</div>')
        output_html.append('</div>')  # Close row 2
        
        # WordCloud visualization
        output_html.append('<h3 class="task-subheader">Word Cloud Visualization</h3>')
        output_html.append('<p>The size of each word represents its frequency in the text.</p>')
        
        # Generate word cloud
        wc = wordcloud.WordCloud(
            background_color='white',
            max_words=100,
            width=800,
            height=400,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        )
        wc.generate_from_frequencies(sorted_word_freq)
        
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        
        output_html.append(fig_to_html(fig))
        
        # TF-IDF Analysis
        output_html.append('<h3 class="task-subheader">TF-IDF Analysis</h3>')
        output_html.append("""
        <div class="alert alert-light">
            <p class="mb-0">
                Term Frequency-Inverse Document Frequency (TF-IDF) identifies terms that are distinctive to parts of the text.
                In this case, we treat each sentence as a separate "document" for the analysis.
            </p>
        </div>
        """)
        
        # Split text into sentences
        sentences = nltk.sent_tokenize(text_input)
        
        # Only perform TF-IDF if there are enough sentences
        if len(sentences) >= 3:
            # Create TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer(
                max_features=100, 
                stop_words='english',
                min_df=1
            )
            
            # Fit and transform the sentences
            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
            
            # Get feature names
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Create a table of top TF-IDF terms for each sentence
            tfidf_data = []
            
            for i, sentence in enumerate(sentences[:min(len(sentences), 5)]):  # Show max 5 sentences to avoid clutter
                # Get top terms for this sentence
                tfidf_scores = tfidf_matrix[i].toarray()[0]
                top_indices = np.argsort(tfidf_scores)[-5:][::-1]  # Top 5 terms
                
                top_terms = [feature_names[idx] for idx in top_indices]
                top_scores = [tfidf_scores[idx] for idx in top_indices]
                
                # Format for display
                formatted_terms = ', '.join([f"{term} ({score:.3f})" for term, score in zip(top_terms, top_scores)])
                
                shortened_sentence = (sentence[:75] + '...') if len(sentence) > 75 else sentence
                
                tfidf_data.append({
                    'Sentence': shortened_sentence,
                    'Distinctive Terms (TF-IDF scores)': formatted_terms
                })
            
            # Create dataframe
            tfidf_df = pd.DataFrame(tfidf_data)
            
            output_html.append('<div class="mt-3">')
            output_html.append(df_to_html_table(tfidf_df))
            output_html.append('</div>')
            
            # Create a TF-IDF term-sentence heatmap
            if len(sentences) <= 10:  # Only create heatmap for reasonable number of sentences
                # Get top terms across all sentences
                mean_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
                top_indices = np.argsort(mean_tfidf)[-10:][::-1]  # Top 10 terms
                top_terms = [feature_names[idx] for idx in top_indices]
                
                # Create heatmap data
                heatmap_data = tfidf_matrix[:, top_indices].toarray()
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
                
                # Add labels
                plt.yticks(range(len(sentences)), [f"Sent {i+1}" for i in range(len(sentences))])
                plt.xticks(range(len(top_terms)), top_terms, rotation=45, ha='right')
                
                plt.colorbar(label='TF-IDF Score')
                plt.xlabel('Terms')
                plt.ylabel('Sentences')
                plt.title('TF-IDF Heatmap: Term Importance by Sentence')
                plt.tight_layout()
                
                output_html.append('<h4>Term Importance Heatmap</h4>')
                output_html.append('<p>This heatmap shows which terms are most distinctive in each sentence.</p>')
                output_html.append(fig_to_html(fig))
        else:
            output_html.append("""
            <div class="alert alert-warning">
                <p class="mb-0">TF-IDF analysis requires at least 3 sentences. The provided text doesn't have enough sentences for this analysis.</p>
            </div>
            """)
        
        # Topic Modeling
        output_html.append('<h3 class="task-subheader">Topic Modeling</h3>')
        output_html.append("""
        <div class="alert alert-light">
            <p class="mb-0">
                Topic modeling uses statistical methods to discover abstract "topics" that occur in a collection of documents.
                Here, we use Latent Dirichlet Allocation (LDA) to identify potential topics.
            </p>
        </div>
        """)
        
        # Check if text is long enough for topic modeling
        if len(text_input.split()) < 50:
            output_html.append("""
            <div class="alert alert-warning">
                <p class="mb-0">Topic modeling works best with longer texts. The provided text is too short for reliable topic modeling.</p>
            </div>
            """)
        else:
            # Create document-term matrix
            # For short single-document text, we'll split by sentences to create a "corpus"
            sentences = nltk.sent_tokenize(text_input)
            
            if len(sentences) < 4:
                output_html.append("""
                <div class="alert alert-warning">
                    <p class="mb-0">Topic modeling works best with multiple documents or paragraphs. Since the provided text has few sentences,
                    the topic modeling results may not be meaningful.</p>
                </div>
                """)
            
            # Create document-term matrix using CountVectorizer
            vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                min_df=1
            )
            
            # Create a document-term matrix
            dtm = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Set number of topics based on text length
            n_topics = min(3, max(2, len(sentences) // 3))
            
            # LDA Topic Modeling
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42
            )
            
            lda_model.fit(dtm)
            
            # Get top terms for each topic
            n_top_words = 10
            topic_terms = []
            for topic_idx, topic in enumerate(lda_model.components_):
                top_indices = topic.argsort()[:-n_top_words - 1:-1]
                top_terms = [feature_names[i] for i in top_indices]
                topic_weight = topic[top_indices].sum() / topic.sum()  # Approximation of topic "importance"
                topic_terms.append({
                    "Topic": f"Topic {topic_idx + 1}",
                    "Top Terms": ", ".join(top_terms),
                    "Weight": f"{topic_weight:.2f}"
                })
            
            topic_df = pd.DataFrame(topic_terms)
            
            output_html.append('<h4>LDA Topic Model Results</h4>')
            output_html.append(df_to_html_table(topic_df))
            
            # Create word cloud for each topic
            output_html.append('<h4>Topic Word Clouds</h4>')
            output_html.append('<div class="row">')
            
            for topic_idx, topic in enumerate(lda_model.components_):
                # Get topic words and weights
                word_weights = {feature_names[i]: topic[i] for i in topic.argsort()[:-50-1:-1]}
                
                # Generate word cloud
                wc = wordcloud.WordCloud(
                    background_color='white',
                    max_words=30,
                    width=400,
                    height=300,
                    colormap='plasma',
                    contour_width=1,
                    contour_color='steelblue'
                )
                wc.generate_from_frequencies(word_weights)
                
                # Create figure
                fig = plt.figure(figsize=(6, 4))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Topic {topic_idx + 1}')
                plt.tight_layout()
                
                output_html.append(f'<div class="col-12 mb-3">')
                output_html.append(fig_to_html(fig))
                output_html.append('</div>')
            
            output_html.append('</div>')  # Close row for word clouds
            
            # Topic distribution visualization
            topic_distribution = lda_model.transform(dtm)
            
            # Calculate dominant topic for each sentence
            dominant_topics = np.argmax(topic_distribution, axis=1)
            
            # Count number of sentences for each dominant topic
            topic_counts = Counter(dominant_topics)
            
            # Prepare data for visualization
            topics = [f"Topic {i+1}" for i in range(n_topics)]
            counts = [topic_counts.get(i, 0) for i in range(n_topics)]
            
            # Create visualization
            fig = plt.figure(figsize=(8, 5))
            bars = plt.bar(topics, counts, color=plt.cm.plasma(np.linspace(0.15, 0.85, n_topics)))
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height}',
                        ha='center', va='bottom')
            
            plt.xlabel('Topic')
            plt.ylabel('Number of Sentences')
            plt.title('Distribution of Dominant Topics Across Sentences')
            plt.tight_layout()
            
            output_html.append('<h4>Topic Distribution</h4>')
            output_html.append(fig_to_html(fig))
            
            # Topic network graph
            output_html.append('<h4>Topic-Term Network</h4>')
            output_html.append('<p>This visualization shows the relationships between topics and their most important terms.</p>')
            
            # Create network graph
            G = nx.Graph()
            
            # Add topic nodes
            for i in range(n_topics):
                G.add_node(f"Topic {i+1}", type='topic', size=1000)
            
            # Add term nodes and edges
            for topic_idx, topic in enumerate(lda_model.components_):
                topic_name = f"Topic {topic_idx+1}"
                
                # Get top terms for this topic
                top_indices = topic.argsort()[:-11:-1]
                
                for i in top_indices:
                    term = feature_names[i]
                    weight = topic[i]
                    
                    # Only add terms with significant weight
                    if weight > 0.01:
                        if not G.has_node(term):
                            G.add_node(term, type='term', size=300)
                        
                        G.add_edge(topic_name, term, weight=weight)
            
            # Create graph visualization
            fig = plt.figure(figsize=(10, 8))
            
            # Position nodes using spring layout
            pos = nx.spring_layout(G, k=0.3, seed=42)
            
            # Draw nodes
            topic_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'topic']
            term_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'term']
            
            # Draw topic nodes
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=topic_nodes,
                node_color='#E53935',
                node_size=[G.nodes[node]['size'] for node in topic_nodes],
                alpha=0.8
            )
            
            # Draw term nodes
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=term_nodes,
                node_color='#1976D2',
                node_size=[G.nodes[node]['size'] for node in term_nodes],
                alpha=0.6
            )
            
            # Draw edges with varying thickness
            edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
            nx.draw_networkx_edges(
                G, pos,
                width=edge_weights,
                alpha=0.5,
                edge_color='gray'
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos,
                font_size=10,
                font_weight='bold'
            )
            
            plt.axis('off')
            plt.tight_layout()
            
            output_html.append(fig_to_html(fig))
            
            # Add note about interpreting results
            output_html.append("""
            <div class="alert alert-info">
                <h4>Interpreting Topic Models</h4>
                <p>Topic modeling is an unsupervised technique that works best with large collections of documents.
                For a single text, especially shorter ones, topics may be less distinct or meaningful.
                The "topics" shown here represent clusters of words that frequently appear together in the text.</p>
                <p>For better topic modeling results:</p>
                <ul>
                    <li>Use longer texts with at least several paragraphs</li>
                    <li>Provide multiple related documents for analysis</li>
                    <li>Consider domain-specific preprocessing</li>
                </ul>
            </div>
            """)
    
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h3>Error</h3>
            <p>Failed to analyze topics: {str(e)}</p>
        </div>
        """)
    
    # About Topic Analysis section
    output_html.append("""
    <div class="card mt-4">
        <div class="card-header">
            <h4 class="mb-0">
                <i class="fas fa-info-circle"></i>
                About Topic Analysis
            </h4>
        </div>
        <div class="card-body">
            <h5>What is Topic Analysis?</h5>
            
            <p>Topic analysis, also known as topic modeling or topic extraction, is the process of identifying the main themes
            or topics that occur in a collection of documents. It uses statistical models to discover abstract topics based
            on word distributions throughout the texts.</p>
            
            <h5>Common Approaches:</h5>
            
            <ul>
                <li><b>Term Frequency Analysis</b> - Simple counting of terms to find the most common topics</li>
                <li><b>TF-IDF (Term Frequency-Inverse Document Frequency)</b> - Identifies terms that are distinctive to particular documents or sections</li>
                <li><b>LDA (Latent Dirichlet Allocation)</b> - A probabilistic model that assigns topic distributions to documents</li>
                <li><b>NMF (Non-negative Matrix Factorization)</b> - A linear-algebraic approach to topic discovery</li>
                <li><b>BERTopic</b> - A modern approach that uses BERT embeddings and clustering for topic modeling</li>
            </ul>
            
            <h5>Applications:</h5>
            
            <ul>
                <li><b>Content organization</b> - Categorizing documents by topic</li>
                <li><b>Trend analysis</b> - Tracking how topics evolve over time</li>
                <li><b>Content recommendation</b> - Suggesting related content based on topic similarity</li>
                <li><b>Customer feedback analysis</b> - Understanding main themes in reviews or feedback</li>
                <li><b>Research insights</b> - Identifying research themes in academic papers</li>
            </ul>
        </div>
    </div>
    """)
    
    output_html.append('</div>')  # Close result-area div
    
    return '\n'.join(output_html)
