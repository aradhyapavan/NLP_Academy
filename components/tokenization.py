import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

from utils.model_loader import load_spacy, download_nltk_resources
from utils.helpers import fig_to_html, df_to_html_table

def tokenization_handler(text_input):
    """Show tokenization capabilities."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Tokenization</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-info-circle"></i>
    Tokenization is the process of breaking text into smaller units called tokens, which can be words, characters, or subwords.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Tools Used:</h4>
        <ul>
            <li><b>NLTK</b> - Natural Language Toolkit for basic word and sentence tokenization</li>
            <li><b>spaCy</b> - Advanced tokenization with linguistic features</li>
            <li><b>WordPiece</b> - Subword tokenization used by BERT and other transformers</li>
        </ul>
    </div>
    """)
    
    try:
        # Ensure NLTK resources are downloaded
        download_nltk_resources()
        
        # Original Text
        output_html.append('<h3 class="task-subheader">Original Text</h3>')
        output_html.append(f'<div class="card"><div class="card-body"><div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 500px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">{text_input}</div></div></div>')
        
        # Word Tokenization
        output_html.append('<h3 class="task-subheader">Word Tokenization</h3>')
        output_html.append('<p>Breaking text into individual words and punctuation marks.</p>')
        
        # NLTK Word Tokenization
        nltk_tokens = word_tokenize(text_input)
        
        # Format tokens
        token_html = ""
        for token in nltk_tokens:
            token_html += f'<span class="token">{token}</span>'
        
        output_html.append(f"""
        <div class="card">
            <div class="card-body">
                <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; line-height: 2.5;">
                    {token_html}
                </div>
            </div>
        </div>
        <style>
            .token {{
                background-color: #E3F2FD;
                border: 1px solid #1976D2;
                border-radius: 4px;
                padding: 3px 6px;
                margin: 3px;
                display: inline-block;
            }}
        </style>
        """)
        
        # Token statistics
        token_count = len(nltk_tokens)
        unique_tokens = len(set([t.lower() for t in nltk_tokens]))
        alpha_only = sum(1 for t in nltk_tokens if t.isalpha())
        numeric = sum(1 for t in nltk_tokens if t.isnumeric())
        punct = sum(1 for t in nltk_tokens if all(c in '.,;:!?-"\'()[]{}' for c in t))
        
        output_html.append(f"""
        <div class="row mt-3">
            <div class="col-md-2">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="text-primary">{token_count}</h5>
                        <small>Total Tokens</small>
                    </div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="text-success">{unique_tokens}</h5>
                        <small>Unique Tokens</small>
                    </div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="text-info">{alpha_only}</h5>
                        <small>Alphabetic</small>
                    </div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="text-warning">{numeric}</h5>
                        <small>Numeric</small>
                    </div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="text-danger">{punct}</h5>
                        <small>Punctuation</small>
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # Sentence Tokenization
        output_html.append('<h3 class="task-subheader">Sentence Tokenization</h3>')
        output_html.append('<p>Dividing text into individual sentences.</p>')
        
        # NLTK Sentence Tokenization
        nltk_sentences = sent_tokenize(text_input)
        
        # Format sentences
        sentence_html = ""
        for i, sentence in enumerate(nltk_sentences):
            sentence_html += f'<div class="sentence"><span class="sentence-num">{i+1}</span> {sentence}</div>'
        
        output_html.append(f"""
        <div class="card">
            <div class="card-body">
                <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
                    {sentence_html}
                </div>
            </div>
        </div>
        <style>
            .sentence {{
                background-color: #E1F5FE;
                border-left: 3px solid #03A9F4;
                padding: 10px;
                margin: 8px 0;
                border-radius: 0 5px 5px 0;
                position: relative;
            }}
            .sentence-num {{
                font-weight: bold;
                color: #0277BD;
                margin-right: 5px;
            }}
        </style>
        """)
        
        output_html.append(f'<p class="mt-3">Text contains {len(nltk_sentences)} sentences with an average of {token_count / len(nltk_sentences):.1f} tokens per sentence.</p>')
        
        # Advanced Tokenization with spaCy
        output_html.append('<h3 class="task-subheader">Linguistic Tokenization (spaCy)</h3>')
        output_html.append('<p>spaCy provides more linguistically-aware tokenization with additional token properties.</p>')
        
        # Load spaCy model
        nlp = load_spacy()
        doc = nlp(text_input)
        
        # Create token table
        token_data = []
        for token in doc:
            token_data.append({
                'Text': token.text,
                'Lemma': token.lemma_,
                'POS': token.pos_,
                'Tag': token.tag_,
                'Dep': token.dep_,
                'Shape': token.shape_,
                'Alpha': token.is_alpha,
                'Stop': token.is_stop
            })
        
        token_df = pd.DataFrame(token_data)
        
        # Display interactive table with expandable rows
        output_html.append("""
        <div class="table-responsive">
            <table class="table table-striped table-hover">
                <thead class="table-primary sticky-top">
                    <tr>
                        <th>Token</th>
                        <th>Lemma</th>
                        <th>POS</th>
                        <th>Tag</th>
                        <th>Dependency</th>
                        <th>Properties</th>
                    </tr>
                </thead>
                <tbody>
        """)
        
        for token in doc:
            # Determine row color based on token type
            row_class = ""
            if token.is_stop:
                row_class = "table-danger"  # Light red for stopwords
            elif token.pos_ == "VERB":
                row_class = "table-success"  # Light green for verbs
            elif token.pos_ == "NOUN" or token.pos_ == "PROPN":
                row_class = "table-primary"  # Light blue for nouns
            elif token.pos_ == "ADJ":
                row_class = "table-warning"  # Light yellow for adjectives
            
            output_html.append(f"""
            <tr class="{row_class}">
                <td><strong>{token.text}</strong></td>
                <td>{token.lemma_}</td>
                <td>{token.pos_}</td>
                <td>{token.tag_}</td>
                <td>{token.dep_}</td>
                <td>
                    <span class="badge {'bg-success' if token.is_alpha else 'bg-danger'}">
                        {'Alpha' if token.is_alpha else 'Non-alpha'}
                    </span>
                    <span class="badge {'bg-danger' if token.is_stop else 'bg-success'}">
                        {'Stopword' if token.is_stop else 'Content'}
                    </span>
                    <span class="badge bg-info">
                        Shape: {token.shape_}
                    </span>
                </td>
            </tr>
            """)
        
        output_html.append("""
                </tbody>
            </table>
        </div>
        """)
        
        # Create visualization for POS distribution
        pos_counts = Counter([token.pos_ for token in doc])
        
        # Create bar chart for POS distribution
        fig = plt.figure(figsize=(10, 6))
        plt.bar(pos_counts.keys(), pos_counts.values(), color='#1976D2')
        plt.xlabel('Part of Speech')
        plt.ylabel('Count')
        plt.title('Part-of-Speech Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_html.append('<h4>Token Distribution by Part of Speech</h4>')
        output_html.append(fig_to_html(fig))
        
        # Subword Tokenization
        output_html.append('<h3 class="task-subheader">Subword Tokenization (WordPiece/BPE)</h3>')
        output_html.append("""
        <div class="alert alert-light">
            <p>
                Subword tokenization breaks words into smaller units to handle rare words and morphologically rich languages.
                This technique is widely used in modern transformer models like BERT, GPT, etc.
            </p>
        </div>
        """)
        
        try:
            from transformers import BertTokenizer, GPT2Tokenizer
            
            # Load tokenizers
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
            # Tokenize with BERT
            bert_tokens = bert_tokenizer.tokenize(text_input)
            
            # Tokenize with GPT-2
            # GPT-2 doesn't have a special tokenize method like BERT, so we encode and decode
            gpt2_encoding = gpt2_tokenizer.encode(text_input)
            gpt2_tokens = [gpt2_tokenizer.decode([token]).strip() for token in gpt2_encoding]
            
            # BERT WordPiece Section
            output_html.append('<h4 class="bg-primary text-white p-3 rounded">BERT WordPiece</h4>')
            output_html.append('<p>BERT uses WordPiece tokenization which marks subword units with ##.</p>')
            
            # Create token display
            output_html.append('<div class="card"><div class="card-body">')
            output_html.append('<div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; line-height: 2.5;">')
            
            for token in bert_tokens:
                if token.startswith("##"):
                    output_html.append(f'<span class="token" style="background-color: #FFECB3; border-color: #FFA000;">{token}</span>')
                else:
                    output_html.append(f'<span class="token">{token}</span>')
            
            output_html.append('</div></div></div>')
            output_html.append(f'<p class="mt-2">Total BERT tokens: {len(bert_tokens)}</p>')
            
            # GPT-2 BPE Section
            output_html.append('<h4 class="bg-primary text-white p-3 rounded mt-4">GPT-2 BPE</h4>')
            output_html.append('<p>GPT-2 uses Byte-Pair Encoding (BPE) tokenization where Ġ represents a space before the token.</p>')
            
            output_html.append('<div class="card"><div class="card-body">')
            output_html.append('<div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; line-height: 2.5;">')
            
            for token in gpt2_tokens:
                if token.startswith("Ġ"):
                    output_html.append(f'<span class="token">{token}</span>')
                else:
                    output_html.append(f'<span class="token" style="background-color: #FFECB3; border-color: #FFA000;">{token}</span>')
            
            output_html.append('</div></div></div>')
            output_html.append(f'<p class="mt-2">Total GPT-2 tokens: {len(gpt2_tokens)}</p>')
            
            # Compare token counts
            output_html.append('<h4>Token Count Comparison</h4>')
            token_count_data = {
                'Tokenizer': ['Words (spaces)', 'NLTK', 'spaCy', 'BERT WordPiece', 'GPT-2 BPE'],
                'Token Count': [
                    len(text_input.split()),
                    len(nltk_tokens),
                    len(doc),
                    len(bert_tokens),
                    len(gpt2_tokens)
                ]
            }
            
            token_count_df = pd.DataFrame(token_count_data)
            
            # Create comparison chart
            fig = plt.figure(figsize=(10, 6))
            bars = plt.bar(token_count_df['Tokenizer'], token_count_df['Token Count'], color=['#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3'])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height}',
                        ha='center', va='bottom')
            
            plt.ylabel('Token Count')
            plt.title('Tokenization Comparison by Method')
            plt.ylim(0, max(token_count_df['Token Count']) * 1.1)  # Add some headroom for labels
            plt.tight_layout()
            
            output_html.append(fig_to_html(fig))
            
            # Add token length distribution analysis
            output_html.append('<h4>Token Length Distribution</h4>')
            token_lengths = [len(token) for token in nltk_tokens]
            
            fig = plt.figure(figsize=(10, 6))
            plt.hist(token_lengths, bins=range(1, max(token_lengths) + 2), color='#4CAF50', alpha=0.7)
            plt.xlabel('Token Length')
            plt.ylabel('Frequency')
            plt.title('Token Length Distribution')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            output_html.append(fig_to_html(fig))
            
            # Add tokenization statistics summary
            avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
            output_html.append(f"""
            <h4>Tokenization Statistics</h4>
            <div class="row mt-3">
                <div class="col-md-4">
                    <div class="card text-center">
                        <div class="card-body">
                            <h3 class="text-success">{token_count}</h3>
                            <p class="mb-0">Total Tokens</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card text-center">
                        <div class="card-body">
                            <h3 class="text-primary">{avg_token_length:.2f}</h3>
                            <p class="mb-0">Average Token Length</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card text-center">
                        <div class="card-body">
                            <h3 class="text-warning">{token_count / len(nltk_sentences):.2f}</h3>
                            <p class="mb-0">Tokens per Sentence</p>
                        </div>
                    </div>
                </div>
            </div>
            """)
            
        except Exception as e:
            output_html.append(f"""
            <div class="alert alert-warning">
                <h4>Subword Tokenization Error</h4>
                <p>Failed to load transformer tokenizers: {str(e)}</p>
                <p>The transformers library may not be installed or there might be network issues when downloading models.</p>
            </div>
            """)
    
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h3>Error</h3>
            <p>Failed to process tokenization: {str(e)}</p>
        </div>
        """)
    
    # About Tokenization section
    output_html.append("""
    <div class="card mt-4">
        <div class="card-header">
            <h4 class="mb-0">
                <i class="fas fa-info-circle"></i>
                About Tokenization
            </h4>
        </div>
        <div class="card-body">
            <h5>What is Tokenization?</h5>
            
            <p>Tokenization is the process of breaking down text into smaller units called tokens.
            These tokens can be words, subwords, characters, or symbols, depending on the approach.
            It's typically the first step in most NLP pipelines.</p>
            
            <h5>Types of Tokenization:</h5>
            
            <ul>
                <li><b>Word Tokenization</b> - Splits text on whitespace and punctuation (with various rules)</li>
                <li><b>Sentence Tokenization</b> - Divides text into sentences using punctuation and other rules</li>
                <li><b>Subword Tokenization</b> - Splits words into meaningful subunits (WordPiece, BPE, SentencePiece)</li>
                <li><b>Character Tokenization</b> - Treats each character as a separate token</li>
            </ul>
            
            <h5>Why Subword Tokenization?</h5>
            
            <p>Modern NLP models use subword tokenization because:</p>
            <ul>
                <li>It handles out-of-vocabulary words better</li>
                <li>It represents rare words by decomposing them</li>
                <li>It works well for morphologically rich languages</li>
                <li>It balances vocabulary size and token length</li>
            </ul>
        </div>
    </div>
    """)
    
    output_html.append('</div>')  # Close result-area div
    
    return '\n'.join(output_html)
