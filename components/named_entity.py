import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import spacy
from collections import Counter
import networkx as nx
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from utils.model_loader import load_spacy
from utils.helpers import fig_to_html, df_to_html_table

def named_entity_handler(text_input):
    """Show named entity recognition capabilities."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Named Entity Recognition</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-info-circle"></i>
    Named Entity Recognition identifies and classifies key information in text into pre-defined categories such as person names, organizations, locations, etc.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Models Used:</h4>
        <ul>
            <li><b>dslim/bert-base-NER</b> - BERT-based Named Entity Recognition model</li>
            <li><b>spaCy en_core_web_sm</b> - Statistical NLP model for additional analysis</li>
            <li><b>Entity Types</b> - Identifies people, organizations, locations, and miscellaneous entities</li>
        </ul>
    </div>
    """)
    
    try:
        # Load BERT NER model
        try:
            tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
            ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        except Exception as model_err:
            output_html.append(f"""
            <div class="alert alert-warning">
                <h4>Model Loading Issue</h4>
                <p>Could not load BERT NER model: {str(model_err)}</p>
                <p>Falling back to spaCy model...</p>
            </div>
            """)
            # Fallback to spaCy
            nlp = load_spacy()
            doc = nlp(text_input)
            bert_entities = []
        else:
            # Process with BERT NER
            bert_entities = ner_pipeline(text_input)
        
        # Also load spaCy for additional analysis
        nlp = load_spacy()
        doc = nlp(text_input)
        
        # Combine entities from both models
        all_entities = []
        
        # Add BERT entities
        for entity in bert_entities:
            all_entities.append({
                'text': entity['word'].replace('##', ''),
                'label': entity['entity_group'],
                'confidence': entity['score'],
                'start': entity['start'],
                'end': entity['end'],
                'source': 'BERT'
            })
        
        # Add spaCy entities
        for ent in doc.ents:
            all_entities.append({
                'text': ent.text,
                'label': ent.label_,
                'confidence': 1.0,  # spaCy doesn't provide confidence scores
                'start': ent.start_char,
                'end': ent.end_char,
                'source': 'spaCy'
            })
        
        # If no entities were found
        if len(all_entities) == 0:
            output_html.append("""
            <div class="alert alert-warning">
                <h3>No Named Entities Found</h3>
                <p>The model couldn't identify any named entities in the provided text. Try a different text that contains names, places, organizations, dates, etc.</p>
            </div>
            """)
        else:
            # Display identified entities in text
            output_html.append('<h3 class="task-subheader">Identified Entities</h3>')
            
            # Color scheme for different entity types (BERT + spaCy)
            colors = {
                # BERT NER labels
                'PER': '#e6194B',         # Person - Red
                'ORG': '#3cb44b',         # Organization - Green  
                'LOC': '#4363d8',         # Location - Blue
                'MISC': '#f58231',        # Miscellaneous - Orange
                # spaCy labels
                'PERSON': '#e6194B',      # Red
                'ORG': '#3cb44b',         # Green
                'GPE': '#4363d8',         # Blue (locations/geopolitical)
                'LOC': '#42d4f4',         # Cyan (non-GPE locations)
                'FACILITY': '#f58231',    # Orange
                'PRODUCT': '#911eb4',     # Purple
                'EVENT': '#f032e6',       # Magenta
                'WORK_OF_ART': '#fabebe', # Pink
                'LAW': '#008080',         # Teal
                'DATE': '#9A6324',        # Brown
                'TIME': '#800000',        # Maroon
                'PERCENT': '#808000',     # Olive
                'MONEY': '#000075',       # Navy
                'QUANTITY': '#000000',    # Black
                'CARDINAL': '#a9a9a9',    # Dark Gray
                'ORDINAL': '#808080',     # Gray
                'NORP': '#469990'         # Nationality/Religious/Political
            }
            
            # Remove duplicates and sort entities by position
            unique_entities = []
            seen_spans = set()
            
            for entity in all_entities:
                span = (entity['start'], entity['end'])
                if span not in seen_spans:
                    unique_entities.append(entity)
                    seen_spans.add(span)
            
            # Sort by start position
            sorted_ents = sorted(unique_entities, key=lambda x: x['start'])
            
            # Create HTML with highlighted entities
            html_text = text_input
            offset = 0
            
            for entity in sorted_ents:
                # Get the appropriate color (default to gray if not found)
                color = colors.get(entity['label'], '#a9a9a9')
                
                # Create the HTML span with tooltip including confidence and source
                start = entity['start'] + offset
                end = entity['end'] + offset
                confidence_text = f" (Confidence: {entity['confidence']:.2f})" if entity['confidence'] < 1.0 else ""
                tooltip = f"{entity['label']} - {entity['source']}{confidence_text}"
                
                entity_html = f'<span class="entity-badge" style="background-color: {color}; color: white; border: 2px solid #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.3);" title="{tooltip}"><strong>{entity["text"]}</strong> <span style="font-size: 0.8em;">({entity["label"]}) ({entity["source"]})</span></span>'
                
                # Replace the entity text with the highlighted version
                html_text = html_text[:start] + entity_html + html_text[end:]
                
                # Update offset for subsequent entities
                offset += len(entity_html) - len(entity['text'])
            
            # Display the highlighted text
            output_html.append(f'<div class="card"><div class="card-body"><div class="entity-text-container">{html_text}</div></div></div>')
            
            # Entity count and distribution
            output_html.append('<h3 class="task-subheader">Entity Distribution</h3>')
            
            # Create a DataFrame for the entities
            entities_data = []
            for entity in unique_entities:
                entities_data.append({
                    'Entity': entity['text'],
                    'Type': entity['label'],
                    'Source': entity['source'],
                    'Confidence': f"{entity['confidence']:.2f}" if entity['confidence'] < 1.0 else "1.00"
                })
            
            entity_df = pd.DataFrame(entities_data)
            
            # Calculate entity type distribution
            entity_counts = Counter([entity['label'] for entity in unique_entities])
            
            # Create bar chart for entity type distribution
            fig = plt.figure(figsize=(12, 8))
            bars = plt.bar(entity_counts.keys(), entity_counts.values(), 
                          color=[colors.get(k, '#a9a9a9') for k in entity_counts.keys()])
            plt.xlabel('Entity Type')
            plt.ylabel('Count')
            plt.title('Entity Type Distribution (BERT + spaCy)')
            plt.xticks(rotation=45, ha='right')
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Chart section
            output_html.append('<section class="entity-chart-section">')
            output_html.append('<div class="chart-container">')
            output_html.append(fig_to_html(fig))
            output_html.append('</div>')
            output_html.append('</section>')
            
            # Table section
            output_html.append('<section class="entity-table-container">')
            output_html.append('<h4>Entities Found</h4>')
            output_html.append(df_to_html_table(entity_df))
            output_html.append('</section>')
            
            # Entity relationship visualization (for texts with multiple entities)
            if len(doc.ents) > 1:
                output_html.append('<h3 class="task-subheader">Entity Relationships</h3>')
                
                # Create a network graph of entities that appear in the same sentence
                G = nx.Graph()
                
                # Add nodes for each unique entity
                for ent in doc.ents:
                    G.add_node(ent.text, type=ent.label_)
                
                # Add edges between entities that appear in the same sentence
                for sent in doc.sents:
                    sent_ents = [ent for ent in doc.ents if sent.start <= ent.start < sent.end]
                    for i, ent1 in enumerate(sent_ents):
                        for ent2 in sent_ents[i+1:]:
                            if G.has_edge(ent1.text, ent2.text):
                                G[ent1.text][ent2.text]['weight'] += 1
                            else:
                                G.add_edge(ent1.text, ent2.text, weight=1)
                
                # Only show relationship visualization if there are edges
                if G.number_of_edges() > 0:
                    # Create a network visualization
                    plt.figure(figsize=(10, 8))
                    
                    # Node colors based on entity type
                    node_colors = [colors.get(G.nodes[node]['type'], '#a9a9a9') for node in G.nodes()]
                    
                    # Position nodes using spring layout
                    pos = nx.spring_layout(G)
                    
                    # Draw the network
                    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, alpha=0.8)
                    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='#888888')
                    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
                    
                    plt.title('Entity Co-occurrence Network')
                    plt.axis('off')
                    plt.tight_layout()
                    
                    output_html.append('<div class="alert alert-light"><p class="mb-0">This visualization shows entities that appear in the same sentences:</p></div>')
                    output_html.append(fig_to_html(plt.gcf()))
                    plt.close()
                else:
                    output_html.append('<p>No entity relationships detected in the text.</p>')
            
            # Legend for entity types
            output_html.append('<h3 class="task-subheader">Entity Type Legend</h3>')
            
            entity_descriptions = {
                'PERSON': 'People, including fictional',
                'ORG': 'Organizations, companies, institutions',
                'GPE': 'Geopolitical entities (countries, cities, states)',
                'LOC': 'Non-GPE locations (mountain ranges, water bodies)',
                'FACILITY': 'Buildings, airports, highways, bridges',
                'PRODUCT': 'Products, objects, vehicles, foods',
                'EVENT': 'Hurricanes, battles, wars, sports events',
                'WORK_OF_ART': 'Titles of books, songs, etc.',
                'LAW': 'Named documents made into laws',
                'DATE': 'Absolute or relative dates',
                'TIME': 'Times smaller than a day',
                'PERCENT': 'Percentage',
                'MONEY': 'Monetary values',
                'QUANTITY': 'Measurements',
                'CARDINAL': 'Numerals not falling under another type',
                'ORDINAL': 'Ordinal numbers',
                'NORP': 'Nationalities, religious or political groups'
            }
            
            output_html.append('<div class="row">')
            for entity, color in colors.items():
                if entity in entity_counts:
                    output_html.append(f"""
                    <div class="col-md-6 mb-2">
                        <div class="card">
                            <div class="card-body p-2">
                                <span class="badge me-2" style="background-color: {color}; color: white;">{entity}</span>
                                <small>{entity_descriptions.get(entity, '')}</small>
                            </div>
                        </div>
                    </div>
                    """)
            output_html.append('</div>')  # Close row
    
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h3>Error</h3>
            <p>Failed to process named entities: {str(e)}</p>
        </div>
        """)
    
    # About NER section
    output_html.append("""
    <div class="card mt-4">
        <div class="card-header">
            <h4 class="mb-0">
                <i class="fas fa-info-circle"></i>
                About Named Entity Recognition
            </h4>
        </div>
        <div class="card-body">
            <h5>What is Named Entity Recognition?</h5>
            
            <p>Named Entity Recognition (NER) is an NLP technique that automatically identifies and classifies named entities 
            in text into predefined categories. These entities are typically proper nouns such as people, organizations, 
            locations, expressions of times, quantities, monetary values, and percentages.</p>
            
            <h5>Applications of NER:</h5>
            
            <ul>
                <li><b>Information Extraction</b> - Identifying key information from large volumes of text</li>
                <li><b>Question Answering</b> - Helping systems understand what entities questions are referring to</li>
                <li><b>Document Classification</b> - Using entity types and frequencies to categorize documents</li>
                <li><b>Customer Service</b> - Identifying product names, issue types, and user information in support tickets</li>
                <li><b>Content Recommendation</b> - Using entities to find related content</li>
            </ul>
        </div>
    </div>
    """)
    
    output_html.append('</div>')  # Close result-area div
    
    return '\n'.join(output_html)
