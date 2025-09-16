import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import Counter

from utils.model_loader import load_sentiment_analyzer, load_emotion_classifier
from utils.helpers import fig_to_html, df_to_html_table

def sentiment_handler(text_input):
    """Show sentiment analysis capabilities."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Sentiment Analysis</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-info-circle"></i>
    Sentiment analysis determines the emotional tone behind text to identify if it expresses positive, negative, or neutral sentiment.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Models Used:</h4>
        <ul>
            <li><b>NLTK VADER</b> - Rule-based sentiment analyzer specifically tuned for social media text</li>
            <li><b>DistilBERT</b> - Transformer model fine-tuned on SST-2 dataset, achieving ~91% accuracy</li>
            <li><b>RoBERTa Emotion</b> - Transformer model for multi-label emotion detection</li>
        </ul>
    </div>
    """)
    
    try:
        # VADER Analysis
        output_html.append('<h3 class="task-subheader">VADER Sentiment Analysis</h3>')
        output_html.append('<p>VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media.</p>')
        
        # Get VADER analyzer
        vader_analyzer = SentimentIntensityAnalyzer()
        vader_scores = vader_analyzer.polarity_scores(text_input)
        
        # Extract scores
        compound_score = vader_scores['compound']
        pos_score = vader_scores['pos']
        neg_score = vader_scores['neg']
        neu_score = vader_scores['neu']
        
        # Determine sentiment category
        if compound_score >= 0.05:
            sentiment_category = "Positive"
            sentiment_color = "#4CAF50"  # Green
            sentiment_emoji = "😊"
        elif compound_score <= -0.05:
            sentiment_category = "Negative"
            sentiment_color = "#F44336"  # Red
            sentiment_emoji = "😞"
        else:
            sentiment_category = "Neutral"
            sentiment_color = "#FFC107"  # Amber
            sentiment_emoji = "😐"
        
        # Create sentiment gauge display
        output_html.append(f"""
        <div class="card">
            <div class="card-body">
                <div class="text-center mb-3">
                    <span style="font-size: 3rem; margin-right: 15px;">{sentiment_emoji}</span>
                    <div>
                        <h3 class="mb-0" style="color: {sentiment_color};">{sentiment_category}</h3>
                        <p class="mb-0 fs-5">Compound Score: {compound_score:.2f}</p>
                    </div>
                </div>
                
                <div style="height: 30px; background-color: #e0e0e0; border-radius: 15px; position: relative; overflow: hidden; margin: 10px 0;">
                    <div style="position: absolute; top: 0; bottom: 0; left: 50%; width: 2px; background-color: #000; z-index: 2;"></div>
                    <div style="position: absolute; top: 0; bottom: 0; left: {(compound_score + 1) / 2 * 100}%; width: 10px; background-color: {sentiment_color}; border-radius: 5px; transform: translateX(-50%); z-index: 3;"></div>
                    <div style="position: absolute; top: 0; bottom: 0; left: 0; width: 50%; background: linear-gradient(90deg, #F44336 0%, #FFC107 100%);"></div>
                    <div style="position: absolute; top: 0; bottom: 0; right: 0; width: 50%; background: linear-gradient(90deg, #FFC107 0%, #4CAF50 100%);"></div>
                </div>
                <div class="d-flex justify-content-between mt-2">
                    <span>Negative (-1.0)</span>
                    <span>Neutral (0.0)</span>
                    <span>Positive (1.0)</span>
                </div>
            </div>
        </div>
        """)
        
        # VADER score breakdown
        output_html.append('<h4>VADER Score Breakdown</h4>')
        
        # Create pie chart
        fig = plt.figure(figsize=(8, 8))
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [pos_score, neu_score, neg_score]
        colors = ['#4CAF50', '#FFC107', '#F44336']
        explode = (0.1, 0, 0) if pos_score > neg_score and pos_score > neu_score else \
                  (0, 0.1, 0) if neu_score > pos_score and neu_score > neg_score else \
                  (0, 0, 0.1)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('VADER Sentiment Distribution')
        
        # Create detail table
        detail_df = pd.DataFrame({
            'Metric': ['Positive Score', 'Neutral Score', 'Negative Score', 'Compound Score'],
            'Value': [pos_score, neu_score, neg_score, compound_score]
        })
        
        # Layout with columns for VADER results
        output_html.append('<div class="row">')
        
        # Column 1: Chart
        output_html.append('<div class="col-md-6">')
        output_html.append(fig_to_html(fig))
        output_html.append('</div>')
        
        # Column 2: Data
        output_html.append('<div class="col-md-6">')
        output_html.append(df_to_html_table(detail_df))
        
        # Add interpretation
        if compound_score >= 0.75:
            interpretation = "Extremely positive sentiment"
        elif compound_score >= 0.5:
            interpretation = "Moderately positive sentiment"
        elif compound_score >= 0.05:
            interpretation = "Slightly positive sentiment"
        elif compound_score > -0.05:
            interpretation = "Neutral sentiment"
        elif compound_score > -0.5:
            interpretation = "Slightly negative sentiment"
        elif compound_score > -0.75:
            interpretation = "Moderately negative sentiment"
        else:
            interpretation = "Extremely negative sentiment"
        
        output_html.append(f"""
        <div class="alert alert-success mt-3">
            <h4>Interpretation</h4>
            <p class="mb-0">{interpretation}</p>
        </div>
        """)
        
        output_html.append('</div>')  # Close column 2
        output_html.append('</div>')  # Close row
        
        # Transformer-based Sentiment Analysis
        output_html.append('<h3 class="task-subheader">Transformer-based Sentiment Analysis</h3>')
        output_html.append('<p>This analysis uses a DistilBERT model fine-tuned on the Stanford Sentiment Treebank dataset.</p>')
        
        try:
            # Load transformer model
            sentiment_model = load_sentiment_analyzer()
            
            # Maximum text length for transformer model (BERT has a 512 token limit)
            max_length = 512
            
            # Get prediction
            truncated_text = text_input[:max_length * 4]  # Rough character estimate
            transformer_result = sentiment_model(truncated_text)
            
            if len(text_input) > max_length * 4:
                output_html.append(f"""
                <div class="alert alert-warning">
                    <p class="mb-0"><b>⚠️ Note:</b> Text was truncated for analysis as it exceeds the model's length limit.</p>
                </div>
                """)
            
            # Extract prediction
            transformer_label = transformer_result[0]['label']
            transformer_score = transformer_result[0]['score']
            
            # Display transformer result
            sentiment_color = "#4CAF50" if transformer_label == "POSITIVE" else "#F44336"
            sentiment_emoji = "😊" if transformer_label == "POSITIVE" else "😞"
            
            output_html.append(f"""
            <div class="card" style="border-color: {sentiment_color};">
                <div class="card-body" style="background-color: {sentiment_color}22;">
                    <div class="d-flex align-items-center">
                        <span style="font-size: 3rem; margin-right: 15px;">{sentiment_emoji}</span>
                        <div>
                            <h3 class="mb-0" style="color: {sentiment_color};">{transformer_label.capitalize()}</h3>
                            <p class="mb-0 fs-5">Confidence: {transformer_score:.2%}</p>
                        </div>
                    </div>
                </div>
            </div>
            """)
            
            # Confidence bar
            output_html.append(f"""
            <div style="height: 30px; background-color: #e0e0e0; border-radius: 15px; position: relative; overflow: hidden; margin: 10px 0;">
                <div style="position: absolute; top: 0; bottom: 0; left: 0; width: {transformer_score * 100}%; background-color: {sentiment_color}; border-radius: 5px;"></div>
                <div style="position: absolute; top: 0; bottom: 0; width: 100%; text-align: center; line-height: 30px; color: #000; font-weight: bold;">
                    {transformer_score:.1%} Confidence
                </div>
            </div>
            """)
        
        except Exception as e:
            output_html.append(f"""
            <div class="alert alert-danger">
                <h4>Transformer Model Error</h4>
                <p>Failed to load or run transformer sentiment model: {str(e)}</p>
                <p>Falling back to VADER results only.</p>
            </div>
            """)
        
        # Emotion Analysis
        output_html.append('<h3 class="task-subheader">Emotion Analysis</h3>')
        output_html.append('<p>Identifying specific emotions in text using a RoBERTa model fine-tuned on the emotion dataset.</p>')
        
        try:
            # Load emotion classifier
            emotion_classifier = load_emotion_classifier()
            
            # Get predictions
            truncated_text = text_input[:max_length * 4]  # Rough character estimate
            emotion_result = emotion_classifier(truncated_text)
            
            # Extract emotion scores
            emotion_scores = {}
            for item in emotion_result[0]:
                emotion_scores[item['label']] = item['score']
            
            # Create emotion dataframe
            emotion_df = pd.DataFrame({
                'Emotion': list(emotion_scores.keys()),
                'Score': list(emotion_scores.values())
            }).sort_values('Score', ascending=False)
            
            # Get primary emotion
            primary_emotion = emotion_df.iloc[0]['Emotion']
            primary_score = emotion_df.iloc[0]['Score']
            
            # Emotion color map
            emotion_colors = {
                'joy': '#FFD54F',
                'anger': '#EF5350',
                'sadness': '#42A5F5',
                'fear': '#9C27B0',
                'surprise': '#26C6DA',
                'love': '#EC407A',
                'disgust': '#66BB6A',
                'optimism': '#FF9800',
                'pessimism': '#795548',
                'trust': '#4CAF50',
                'anticipation': '#FF7043',
                'neutral': '#9E9E9E'
            }
            
            # Emotion emoji map
            emotion_emojis = {
                'joy': '😃',
                'anger': '😠',
                'sadness': '😢',
                'fear': '😨',
                'surprise': '😲',
                'love': '❤️',
                'disgust': '🤢',
                'optimism': '🤩',
                'pessimism': '😒',
                'trust': '🤝',
                'anticipation': '🤔',
                'neutral': '😐'
            }
            
            # Create bar chart
            fig = plt.figure(figsize=(10, 6))
            bars = plt.barh(
                emotion_df['Emotion'], 
                emotion_df['Score'], 
                color=[emotion_colors.get(emotion, '#9E9E9E') for emotion in emotion_df['Emotion']]
            )
            plt.xlabel('Score')
            plt.title('Emotion Scores')
            
            # Add value labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f"{bar.get_width():.2f}", va='center')
            
            plt.xlim(0, 1)
            plt.tight_layout()
            
            # Chart section
            output_html.append('<section class="emotion-chart-section">')
            output_html.append('<div class="chart-container">')
            output_html.append(fig_to_html(fig))
            output_html.append('</div>')
            output_html.append('</section>')
            
            # Primary emotion section
            primary_color = emotion_colors.get(primary_emotion, '#9E9E9E')
            primary_emoji = emotion_emojis.get(primary_emotion, '😐')
            
            output_html.append('<section class="emotion-result-container">')
            output_html.append(f"""
            <div class="card" style="border-color: {primary_color};">
                <div class="card-body" style="background-color: {primary_color}22;">
                    <div class="d-flex align-items-center">
                        <span style="font-size: 3rem; margin-right: 15px;">{primary_emoji}</span>
                        <div>
                            <h3 class="mb-0" style="color: {primary_color};">{primary_emotion.capitalize()}</h3>
                            <p class="mb-0 fs-5">Score: {primary_score:.2f}</p>
                        </div>
                    </div>
                </div>
            </div>
            """)
            
            # Show top emotions table
            output_html.append('<h4>Top Emotions</h4>')
            output_html.append(df_to_html_table(emotion_df.head(5)))
            output_html.append('</section>')  # Close emotion result container
        
        except Exception as e:
            output_html.append(f"""
            <div class="alert alert-danger">
                <h4>Emotion Analysis Error</h4>
                <p>Failed to load or run emotion classifier: {str(e)}</p>
            </div>
            """)
        
        # Sentence-level Analysis
        output_html.append('<h3 class="task-subheader">Sentence-level Analysis</h3>')
        output_html.append('<p>Breaking down sentiment by individual sentences to identify sentiment variations throughout the text.</p>')
        
        # Split text into sentences
        sentences = nltk.sent_tokenize(text_input)
        
        # Minimum 2 sentences to do the analysis
        if len(sentences) >= 2:
            # Calculate sentiment for each sentence
            sentence_sentiments = []
            for i, sentence in enumerate(sentences):
                vader_score = vader_analyzer.polarity_scores(sentence)
                sentence_sentiments.append({
                    'Sentence': sentence,
                    'Index': i + 1,
                    'Compound': vader_score['compound'],
                    'Positive': vader_score['pos'],
                    'Negative': vader_score['neg'],
                    'Neutral': vader_score['neu'],
                    'Sentiment': 'Positive' if vader_score['compound'] >= 0.05 else 'Negative' if vader_score['compound'] <= -0.05 else 'Neutral'
                })
            
            # Create DataFrame
            sent_df = pd.DataFrame(sentence_sentiments)
            
            # Create line graph of sentiment flow
            fig = plt.figure(figsize=(10, 6))
            plt.plot(sent_df['Index'], sent_df['Compound'], 'o-', color='#1976D2', linewidth=2, markersize=8)
            plt.axhline(y=0, color='#9E9E9E', linestyle='-', alpha=0.3)
            plt.axhline(y=0.05, color='#4CAF50', linestyle='--', alpha=0.3)
            plt.axhline(y=-0.05, color='#F44336', linestyle='--', alpha=0.3)
            
            # Annotate with sentiment
            for i, row in sent_df.iterrows():
                if row['Sentiment'] == 'Positive':
                    color = '#4CAF50'
                elif row['Sentiment'] == 'Negative':
                    color = '#F44336'
                else:
                    color = '#9E9E9E'
                    
                plt.scatter(row['Index'], row['Compound'], color=color, s=100, zorder=5)
            
            plt.grid(alpha=0.3)
            plt.xlabel('Sentence Number')
            plt.ylabel('Compound Sentiment Score')
            plt.title('Sentiment Flow Through Text')
            plt.ylim(-1.05, 1.05)
            plt.tight_layout()
            
            # Calculate statistics
            positive_count = sum(1 for score in sent_df['Compound'] if score >= 0.05)
            negative_count = sum(1 for score in sent_df['Compound'] if score <= -0.05)
            neutral_count = len(sent_df) - positive_count - negative_count
            
            # Chart section
            output_html.append('<section class="sentence-chart-section">')
            output_html.append('<div class="chart-container">')
            output_html.append(fig_to_html(fig))
            output_html.append('</div>')
            output_html.append('</section>')
            
            # Sentence analysis section
            output_html.append('<section class="sentence-analysis-container">')
            
            # Create sentence stats
            output_html.append(f"""
            <div class="row mb-3">
                <div class="col-4">
                    <div class="card text-center">
                        <div class="card-body p-2">
                            <h5 class="text-success">{positive_count}</h5>
                            <small>Positive</small>
                        </div>
                    </div>
                </div>
                <div class="col-4">
                    <div class="card text-center">
                        <div class="card-body p-2">
                            <h5 class="text-warning">{neutral_count}</h5>
                            <small>Neutral</small>
                        </div>
                    </div>
                </div>
                <div class="col-4">
                    <div class="card text-center">
                        <div class="card-body p-2">
                            <h5 class="text-danger">{negative_count}</h5>
                            <small>Negative</small>
                        </div>
                    </div>
                </div>
            </div>
            """)
            
            # Display sentiment swings
            sentiment_changes = 0
            prev_sentiment = None
            for sentiment in sent_df['Sentiment']:
                if prev_sentiment is not None and sentiment != prev_sentiment:
                    sentiment_changes += 1
                prev_sentiment = sentiment
            
            if sentiment_changes > 0:
                output_html.append(f"""
                <div class="alert alert-success">
                    <p class="mb-0"><b>Sentiment Shifts:</b> {sentiment_changes}</p>
                    <p class="mb-0">The text shows {sentiment_changes} shifts in sentiment between sentences.</p>
                </div>
                """)
            
            # Show sentence breakdown table
            output_html.append('<h4>Sentence-by-Sentence Analysis</h4>')
            
            # Custom HTML table for better formatting
            output_html.append('<div class="table-responsive" style="max-height: 400px;">')
            output_html.append('<table class="table table-striped">')
            output_html.append('<thead><tr><th>#</th><th>Sentence</th><th>Sentiment</th></tr></thead>')
            output_html.append('<tbody>')
            
            for i, row in sent_df.iterrows():
                if row['Sentiment'] == 'Positive':
                    bg_class = 'table-success'
                    sentiment_html = f"""
                    <div class="d-flex align-items-center">
                        <span class="me-2">😊</span>
                        <span class="text-success fw-bold">Positive</span>
                        <span class="ms-2 text-muted">({row['Compound']:.2f})</span>
                    </div>
                    """
                elif row['Sentiment'] == 'Negative':
                    bg_class = 'table-danger'
                    sentiment_html = f"""
                    <div class="d-flex align-items-center">
                        <span class="me-2">😞</span>
                        <span class="text-danger fw-bold">Negative</span>
                        <span class="ms-2 text-muted">({row['Compound']:.2f})</span>
                    </div>
                    """
                else:
                    bg_class = 'table-warning'
                    sentiment_html = f"""
                    <div class="d-flex align-items-center">
                        <span class="me-2">😐</span>
                        <span class="text-warning fw-bold">Neutral</span>
                        <span class="ms-2 text-muted">({row['Compound']:.2f})</span>
                    </div>
                    """
                
                output_html.append(f'<tr class="{bg_class}">')
                output_html.append(f'<td>{i+1}</td>')
                output_html.append(f'<td>{row["Sentence"]}</td>')
                output_html.append(f'<td>{sentiment_html}</td>')
                output_html.append('</tr>')
            
            output_html.append('</tbody></table>')
            output_html.append('</div>')
            output_html.append('</section>')  # Close sentence analysis container
        else:
            output_html.append("""
            <div class="alert alert-warning">
                <p class="mb-0">Sentence-level analysis requires at least two sentences. The provided text doesn't have enough sentences for this analysis.</p>
            </div>
            """)
    
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h3>Error</h3>
            <p>Failed to analyze sentiment: {str(e)}</p>
        </div>
        """)
    
    # About Sentiment Analysis section
    output_html.append("""
    <div class="card mt-4">
        <div class="card-header">
            <h4 class="mb-0">
                <i class="fas fa-info-circle"></i>
                About Sentiment Analysis
            </h4>
        </div>
        <div class="card-body">
            <h5>What is Sentiment Analysis?</h5>
            
            <p>Sentiment Analysis (also known as opinion mining) is a natural language processing technique that identifies
            and extracts subjective information from text. It determines whether a piece of text expresses positive, negative,
            or neutral sentiment.</p>
            
            <h5>Common Approaches:</h5>
            
            <ol>
                <li><b>Lexicon-based</b> (like VADER) - Uses dictionaries of words with pre-assigned sentiment scores</li>
                <li><b>Machine learning</b> - Supervised techniques that learn from labeled data</li>
                <li><b>Deep learning</b> (like our Transformer models) - Neural networks that can capture complex patterns and contexts</li>
            </ol>
            
            <h5>Applications:</h5>
            
            <ul>
                <li><b>Brand monitoring</b> - Track public perception of a brand</li>
                <li><b>Customer feedback analysis</b> - Understand customer satisfaction</li>
                <li><b>Market research</b> - Analyze product reviews and consumer opinions</li>
                <li><b>Social media monitoring</b> - Track public sentiment on topics or events</li>
                <li><b>Stock market prediction</b> - Analyze news sentiment to predict stock movements</li>
            </ul>
        </div>
    </div>
    """)
    
    output_html.append('</div>')  # Close result-area div
    
    return '\n'.join(output_html)
