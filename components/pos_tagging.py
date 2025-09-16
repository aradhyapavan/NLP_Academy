import matplotlib.pyplot as plt
import pandas as pd
import nltk
from collections import Counter
import random
import numpy as np
import io
import base64
from PIL import Image

from utils.model_loader import load_spacy
from utils.helpers import fig_to_html, df_to_html_table, format_pos_token

def pos_tagging_handler(text_input):
    """Show part-of-speech tagging capabilities."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Part-of-Speech Tagging</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-info-circle"></i>
    Part-of-Speech (POS) tagging is the process of marking up words in text according to their grammatical categories
    such as noun, verb, adjective, etc.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Models Used:</h4>
        <ul>
            <li><b>NLTK</b> - Using the Perceptron tagger trained on the Penn Treebank corpus</li>
            <li><b>spaCy</b> - Using the en_core_web_sm model's POS tagging capabilities</li>
        </ul>
    </div>
    """)
    
    try:
        # Process with NLTK
        words = nltk.word_tokenize(text_input)
        nltk_pos = nltk.pos_tag(words)
        
        # Process with spaCy
        nlp = load_spacy()
        doc = nlp(text_input)
        spacy_pos = [(token.text, token.pos_) for token in doc]
        
        # Display tagged text
        output_html.append('<h3 class="task-subheader">Tagged Text</h3>')
        
        # Color scheme for different POS tags
        # Using a visually distinct color palette
        colors = {
            # NLTK Penn Treebank Tags
            'NN': '#e6194B',     # Noun - Red
            'NNS': '#e6194B',    # Plural noun - Red
            'NNP': '#3cb44b',    # Proper noun - Green
            'NNPS': '#3cb44b',   # Plural proper noun - Green
            'VB': '#4363d8',     # Verb - Blue
            'VBD': '#4363d8',    # Verb, past tense - Blue
            'VBG': '#4363d8',    # Verb, gerund - Blue
            'VBN': '#4363d8',    # Verb, past participle - Blue
            'VBP': '#4363d8',    # Verb, non-3rd singular present - Blue
            'VBZ': '#4363d8',    # Verb, 3rd singular present - Blue
            'JJ': '#f58231',     # Adjective - Orange
            'JJR': '#f58231',    # Comparative adjective - Orange
            'JJS': '#f58231',    # Superlative adjective - Orange
            'RB': '#911eb4',     # Adverb - Purple
            'RBR': '#911eb4',    # Comparative adverb - Purple
            'RBS': '#911eb4',    # Superlative adverb - Purple
            'IN': '#f032e6',     # Preposition - Magenta
            'DT': '#fabebe',     # Determiner - Pink
            'PRP': '#008080',    # Personal pronoun - Teal
            'PRP$': '#008080',   # Possessive pronoun - Teal
            'CC': '#9A6324',     # Coordinating conjunction - Brown
            'CD': '#800000',     # Cardinal number - Maroon
            'EX': '#808000',     # Existential there - Olive
            'FW': '#000075',     # Foreign word - Navy
            'MD': '#a9a9a9',     # Modal - Dark Gray
            'PDT': '#469990',    # Predeterminer - Greenish
            'POS': '#000000',    # Possessive ending - Black
            'RP': '#aaffc3',     # Particle - Mint
            'SYM': '#ffd8b1',    # Symbol - Light Orange
            'TO': '#fffac8',     # to - Light Yellow
            'UH': '#dcbeff',     # Interjection - Lavender
            'WDT': '#808080',    # Wh-determiner - Gray
            'WP': '#808080',     # Wh-pronoun - Gray
            'WP$': '#808080',    # Possessive wh-pronoun - Gray
            'WRB': '#808080',    # Wh-adverb - Gray
            
            # spaCy Universal POS Tags
            'NOUN': '#e6194B',   # Noun - Red
            'PROPN': '#3cb44b',  # Proper noun - Green
            'VERB': '#4363d8',   # Verb - Blue
            'ADJ': '#f58231',    # Adjective - Orange
            'ADV': '#911eb4',    # Adverb - Purple
            'ADP': '#f032e6',    # Adposition (preposition) - Magenta
            'DET': '#fabebe',    # Determiner - Pink
            'PRON': '#008080',   # Pronoun - Teal
            'CCONJ': '#9A6324',  # Coordinating conjunction - Brown
            'NUM': '#800000',    # Numeral - Maroon
            'PART': '#aaffc3',   # Particle - Mint
            'INTJ': '#dcbeff',   # Interjection - Lavender
            'PUNCT': '#000000',  # Punctuation - Black
            'SYM': '#ffd8b1',    # Symbol - Light Orange
            'X': '#808080',      # Other - Gray
            'SPACE': '#ffffff'   # Space - White
        }
        
        # Function to generate HTML for POS tagged text
        def generate_tagged_html(pos_tags, tagset_name):
            html = '<div style="line-height: 2.5; padding: 15px; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 20px; overflow-wrap: break-word; word-wrap: break-word;">'
            
            for word, tag in pos_tags:
                # Skip pure whitespace tokens
                if word.strip() == '':
                    html += ' '
                    continue
                    
                # Get color (default to gray if tag not in colors)
                color = colors.get(tag, '#a9a9a9')
                
                # Add tooltip with tag and make sure tags wrap properly
                html += f'<span style="background-color: {color}; color: white; padding: 2px 4px; margin: 2px; border-radius: 4px; display: inline-block;" title="{tag}">{word}</span>'
            
            html += '</div>'
            return html
        
        # Display NLTK and spaCy in a row, one after another
        output_html.append('<div class="row">')
        
        # NLTK Section
        output_html.append('<div class="col-md-6">')
        output_html.append('<div class="card">')
        output_html.append('<div class="card-header">')
        output_html.append('<h4 class="mb-0 text-primary">NLTK (Penn Treebank)</h4>')
        output_html.append('</div>')
        output_html.append('<div class="card-body">')
        output_html.append(generate_tagged_html(nltk_pos, "Penn Treebank"))
        output_html.append('</div>')
        output_html.append('</div>')
        output_html.append('</div>')
        
        # spaCy Section
        output_html.append('<div class="col-md-6">')
        output_html.append('<div class="card">')
        output_html.append('<div class="card-header">')
        output_html.append('<h4 class="mb-0 text-primary">spaCy (Universal)</h4>')
        output_html.append('</div>')
        output_html.append('<div class="card-body">')
        output_html.append(generate_tagged_html(spacy_pos, "Universal"))
        output_html.append('</div>')
        output_html.append('</div>')
        output_html.append('</div>')
        
        output_html.append('</div>')  # Close the row
        
        # Syntactic Tree Visualization (Dependency Parse)
        output_html.append('<h3 class="task-subheader">Sentence Structure Visualization</h3>')
        
        # Split visualizations for each sentence to avoid overcrowding
        sentences = list(doc.sents)
        
        if not sentences:
            output_html.append('<p>No complete sentences found for visualization.</p>')
        else:
            # Add description for dependency parsing
            output_html.append("""
            <div class="alert alert-light">
                <p class="mb-0">
                These diagrams show the grammatical structure of each sentence. 
                Words are connected with arrows that represent the syntactic relationships between them.
                </p>
            </div>
            """)
            
            # For each sentence, create a dependency visualization
            for i, sent in enumerate(sentences):
                if len(sent) > 50:  # Skip very long sentences that might break the visualization
                    output_html.append(f'<div class="alert alert-warning"><strong>Note:</strong> Sentence {i+1} is too long ({len(sent)} tokens) for visualization.</div>')
                    continue
                
                # Create the sentence dependency visualization using matplotlib
                try:
                    # Try to generate the dependency visualization
                    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
                    # Clear the axes before drawing
                    ax.clear()
                    
                    # Draw connecting arcs between words
                    words = [token.text for token in sent]
                    positions = list(range(len(words)))
                    
                    # Draw words
                    for i, word in enumerate(words):
                        ax.text(i, 0, word, ha='center')
                    
                    # Draw arcs for dependencies
                    max_height = 1
                    for token in sent:
                        if token.dep_ and token.head.i != token.i:  # Skip root dependency
                            # Determine start and end positions
                            start = token.i - sent.start
                            end = token.head.i - sent.start
                            
                            # Make sure start is before end
                            if start > end:
                                start, end = end, start
                            
                            # Determine the height of the arc (based on distance)
                            height = 0.2 + (end - start) * 0.1
                            max_height = max(max_height, height + 0.3)
                            
                            # Draw the dependency arc
                            arc_xs = np.linspace(start, end, 50)
                            arc_ys = [height * np.sin((x - start) / (end - start) * np.pi) for x in arc_xs]
                            ax.plot(arc_xs, arc_ys, color=colors.get(token.pos_, 'gray'), lw=1.5)
                            
                            # Add dependency label at the peak of the arc
                            mid_point = (start + end) / 2
                            label_height = height * 0.95  # Just below the peak
                            ax.text(mid_point, label_height, token.dep_, ha='center', fontsize=8, 
                                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
                    
                    # Set axis limits
                    ax.set_xlim([-0.5, len(words) - 0.5])
                    ax.set_ylim([0, max_height + 0.2])
                    
                    # Remove axes and set title
                    ax.axis('off')
                    plt.tight_layout()
                    
                    # Render the plot to HTML
                    output_html.append(fig_to_html(fig))
                    plt.close(fig)
                    
                except Exception as viz_err:
                    output_html.append(f'<div class="alert alert-danger"><strong>Error:</strong> Failed to visualize sentence {i+1}: {str(viz_err)}</div>')
        
        # POS Distribution Analysis
        output_html.append('<h3 class="task-subheader">POS Distribution Analysis</h3>')
        
        # Calculate POS distribution using spaCy tags (more consistent)
        pos_counts = Counter([token.pos_ for token in doc])
        
        # Create bar chart for POS distribution
        fig = plt.figure(figsize=(10, 6))
        bars = plt.bar(pos_counts.keys(), pos_counts.values(), color=[colors.get(k, '#a9a9a9') for k in pos_counts.keys()])
        plt.xlabel('Part of Speech')
        plt.ylabel('Count')
        plt.title('Part-of-Speech Distribution')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Chart section
        output_html.append('<section class="pos-chart-section">')
        output_html.append('<div class="chart-container">')
        output_html.append(fig_to_html(fig))
        output_html.append('</div>')
        output_html.append('</section>')
        
        # Table section
        output_html.append('<section class="pos-table-container">')
        output_html.append('<div class="row">')
        output_html.append('<div class="col-md-6">')
        
        # Create a DataFrame for the POS counts
        pos_df = pd.DataFrame({
            'POS Tag': list(pos_counts.keys()),
            'Count': list(pos_counts.values()),
            'Percentage': [count/sum(pos_counts.values())*100 for count in pos_counts.values()]
        })
        pos_df = pos_df.sort_values('Count', ascending=False).reset_index(drop=True)
        
        # Add percentage column
        pos_df['Percentage'] = pos_df['Percentage'].map('{:.1f}%'.format)
        
        output_html.append(df_to_html_table(pos_df))
        output_html.append('</div>')
        
        # Most common words section
        output_html.append('<div class="col-md-6">')
        output_html.append('<h4 class="mt-0">Most Common Words by POS</h4>')
        
        # Get common words for major POS categories
        major_pos = ['NOUN', 'VERB', 'ADJ', 'ADV']
        common_words = {}
        
        for pos in major_pos:
            words = [token.text.lower() for token in doc if token.pos_ == pos]
            if words:
                word_counts = Counter(words).most_common(5)
                common_words[pos] = word_counts
        
        # Create HTML for common words
        for pos, words in common_words.items():
            if words:
                output_html.append(f'<h5>{pos}</h5>')
                output_html.append('<div class="d-flex flex-wrap gap-1 mb-2">')
                
                for word, count in words:
                    # Get appropriate color
                    color = colors.get(pos, '#a9a9a9')
                    output_html.append(f'<span class="badge" style="background-color: {color}; color: white;">{word} ({count})</span>')
                
                output_html.append('</div>')
        
        output_html.append('</div>')  # Close column 2
        output_html.append('</div>')  # Close row
        output_html.append('</section>')  # Close table section
        
        # Add Sentence Grammatical Analysis
        output_html.append('<h3 class="task-subheader">Grammatical Analysis</h3>')
        output_html.append('<p>Detailed analysis of the grammatical components in each sentence.</p>')
        
        # Create Grammatical Role Table
        grammatical_roles = []
        for token in doc:
            if token.dep_ not in ["punct", "space"]:  # Skip punctuation and spaces
                grammatical_roles.append({
                    "Word": token.text,
                    "POS": token.pos_,
                    "Dependency": token.dep_,
                    "Head": token.head.text,
                    "Description": get_dependency_description(token.dep_)
                })
        
        # Convert to DataFrame
        if grammatical_roles:
            roles_df = pd.DataFrame(grammatical_roles)
            output_html.append('<div class="table-responsive" style="max-height: 400px;">')
            output_html.append(df_to_html_table(roles_df))
            output_html.append('</div>')
        else:
            output_html.append('<p>No grammatical roles found to analyze.</p>')
        
        # POS Tag Legend
        output_html.append('<h3 class="task-subheader">POS Tag Legend</h3>')
        
        # Create button toggle for different tagsets
        output_html.append('<div class="card">')
        output_html.append('<div class="card-header text-center">')
        output_html.append('<div class="btn-group pos-legend-buttons" role="group" aria-label="POS Tag Types">')
        output_html.append('<button type="button" class="btn btn-primary btn-lg active" id="universal-btn" onclick="showPOSTags(\'universal\')">Universal Tags</button>')
        output_html.append('<button type="button" class="btn btn-outline-primary btn-lg" id="penn-btn" onclick="showPOSTags(\'penn\')">Penn Treebank Tags</button>')
        output_html.append('</div>')
        output_html.append('</div>')
        output_html.append('<div class="card-body">')
        output_html.append('<div id="pos-content">')
        
        # Universal Tags
        output_html.append('<div class="pos-tags-section" id="universal-tags" style="display: block;">')
        
        universal_tags = {
            'NOUN': 'Nouns - people, places, things',
            'PROPN': 'Proper nouns - specific named entities',
            'VERB': 'Verbs - actions, occurrences',
            'ADJ': 'Adjectives - describe nouns',
            'ADV': 'Adverbs - modify verbs, adjectives, or other adverbs',
            'ADP': 'Adpositions - prepositions, postpositions',
            'DET': 'Determiners - articles and other noun modifiers',
            'PRON': 'Pronouns - words that substitute for nouns',
            'CCONJ': 'Coordinating conjunctions - connect words, phrases, clauses',
            'SCONJ': 'Subordinating conjunctions - connect clauses',
            'NUM': 'Numerals - numbers',
            'PART': 'Particles - function words associated with another word',
            'INTJ': 'Interjections - exclamatory words',
            'PUNCT': 'Punctuation',
            'SYM': 'Symbols',
            'X': 'Other - foreign words, typos, abbreviations',
            'SPACE': 'Space - white spaces'
        }
        
        output_html.append('<div class="row">')
        
        for tag, description in universal_tags.items():
            if tag in colors:
                output_html.append(f"""
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center p-2 border rounded">
                        <span class="badge me-2" style="background-color: {colors[tag]}; color: white; min-width: 60px;">{tag}</span>
                        <span class="small">{description}</span>
                    </div>
                </div>
                """)
        
        output_html.append('</div>')  # Close row
        output_html.append('</div>')  # Close universal tags tab
        
        # Penn Treebank Tags  
        output_html.append('<div class="pos-tags-section" id="penn-tags" style="display: none;">')
        
        penn_tags = {
            'CC': 'Coordinating conjunction',
            'CD': 'Cardinal number',
            'DT': 'Determiner',
            'EX': 'Existential there',
            'FW': 'Foreign word',
            'IN': 'Preposition or subordinating conjunction',
            'JJ': 'Adjective',
            'JJR': 'Adjective, comparative',
            'JJS': 'Adjective, superlative',
            'LS': 'List item marker',
            'MD': 'Modal',
            'NN': 'Noun, singular or mass',
            'NNS': 'Noun, plural',
            'NNP': 'Proper noun, singular',
            'NNPS': 'Proper noun, plural',
            'PDT': 'Predeterminer',
            'POS': 'Possessive ending',
            'PRP': 'Personal pronoun',
            'PRP$': 'Possessive pronoun',
            'RB': 'Adverb',
            'RBR': 'Adverb, comparative',
            'RBS': 'Adverb, superlative',
            'RP': 'Particle',
            'SYM': 'Symbol',
            'TO': 'to',
            'UH': 'Interjection',
            'VB': 'Verb, base form',
            'VBD': 'Verb, past tense',
            'VBG': 'Verb, gerund or present participle',
            'VBN': 'Verb, past participle',
            'VBP': 'Verb, non-3rd person singular present',
            'VBZ': 'Verb, 3rd person singular present',
            'WDT': 'Wh-determiner',
            'WP': 'Wh-pronoun',
            'WP$': 'Possessive wh-pronoun',
            'WRB': 'Wh-adverb'
        }
        
        output_html.append('<div class="row">')
        
        for tag, description in penn_tags.items():
            if tag in colors:
                output_html.append(f"""
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center p-2 border rounded">
                        <span class="badge me-2" style="background-color: {colors[tag]}; color: white; min-width: 60px;">{tag}</span>
                        <span class="small">{description}</span>
                    </div>
                </div>
                """)
        
        output_html.append('</div>')  # Close row
        output_html.append('</div>')  # Close penn tags section
        output_html.append('</div>')  # Close pos content
        output_html.append('</div>')  # Close card body
        output_html.append('</div>')  # Close card
    
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h3>Error</h3>
            <p>Failed to process part-of-speech tagging: {str(e)}</p>
        </div>
        """)
    
    # About POS Tagging section
    output_html.append("""
    <div class="card mt-4">
        <div class="card-header">
            <h4 class="mb-0">
                <i class="fas fa-info-circle"></i>
                About Part-of-Speech Tagging
            </h4>
        </div>
        <div class="card-body">
            <h5>What is Part-of-Speech Tagging?</h5>
            
            <p>Part-of-Speech (POS) tagging is the process of assigning grammatical categories (such as noun, verb, adjective, etc.) 
            to each word in a text. It's one of the fundamental steps in natural language processing.</p>
            
            <h5>Why is POS Tagging Important?</h5>
            
            <ol>
                <li><b>Disambiguation</b> - Words can have multiple meanings depending on their usage. POS tags help disambiguate words.</li>
                <li><b>Syntactic Parsing</b> - POS tags form the basis for higher-level syntactic analysis.</li>
                <li><b>Named Entity Recognition</b> - POS tags help in identifying entities.</li>
                <li><b>Information Extraction</b> - They help in extracting specific information from text.</li>
                <li><b>Text-to-Speech Systems</b> - For correct pronunciation based on word function.</li>
            </ol>
            
            <h5>Tagsets:</h5>
            
            <ul>
                <li><b>Universal Tagset</b> - A simpler, cross-linguistic set with about 17 tags.</li>
                <li><b>Penn Treebank</b> - A more detailed English-specific tagset with about 36 tags.</li>
            </ul>
        </div>
    </div>
    """)
    
    output_html.append('</div>')  # Close result-area div
    
    return '\n'.join(output_html)

def get_dependency_description(dep_tag):
    """Return a description for common dependency tags"""
    descriptions = {
        "ROOT": "Root of the sentence",
        "nsubj": "Nominal subject",
        "obj": "Direct object",
        "dobj": "Direct object",
        "iobj": "Indirect object",
        "det": "Determiner",
        "amod": "Adjectival modifier",
        "advmod": "Adverbial modifier",
        "pobj": "Object of preposition",
        "prep": "Preposition",
        "aux": "Auxiliary verb",
        "cc": "Coordinating conjunction",
        "conj": "Conjunct",
        "mark": "Marker",
        "nmod": "Nominal modifier",
        "compound": "Compound word",
        "attr": "Attribute",
        "case": "Case marker",
        "neg": "Negation modifier",
        "punct": "Punctuation",
        "nsubjpass": "Passive nominal subject",
        "auxpass": "Passive auxiliary",
        "ccomp": "Clausal complement",
        "xcomp": "Open clausal complement",
        "acl": "Adjectival clause",
        "advcl": "Adverbial clause modifier",
        "relcl": "Relative clause modifier"
    }
    return descriptions.get(dep_tag, "Dependency relation")
