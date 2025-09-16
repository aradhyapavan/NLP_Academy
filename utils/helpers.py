import base64
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import plotly.graph_objects as go
import nltk

def fig_to_html(fig, width=None):
    """Convert a matplotlib figure to HTML with optional responsive width"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    
    # Add style attribute if width is specified
    style_attr = ""
    if width:
        style_attr = f' style="width: {width}; max-width: 100%;"'
    
    return f'<img{style_attr} src="data:image/png;base64,{b64}" alt="Plot">'

def df_to_html_table(df):
    """Convert a pandas dataframe to an HTML table with Bootstrap styling"""
    return df.to_html(index=False, classes='table table-striped table-hover', escape=False, table_id='data-table')

def text_statistics(text):
    """Calculate basic text statistics"""
    if not text:
        return {"chars": 0, "words": 0, "sentences": 0}
    
    word_count = len(text.split())
    char_count = len(text)
    
    try:
        sentence_count = len(nltk.sent_tokenize(text))
    except:
        sentence_count = 0
    
    return {"chars": char_count, "words": word_count, "sentences": sentence_count}

def create_text_length_chart(text):
    """Create chart showing text length metrics."""
    words = text.split()
    sentences = nltk.sent_tokenize(text)
    chars = len(text)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Characters', 'Words', 'Sentences'],
        y=[chars, len(words), len(sentences)],
        marker_color=['#90CAF9', '#1E88E5', '#0D47A1']
    ))
    
    fig.update_layout(
        title="Text Length Metrics",
        xaxis_title="Metric",
        yaxis_title="Count",
        template="plotly_white",
        height=400
    )
    
    return fig

def get_image_download_link(fig, filename, text):
    """Generate an HTML representation of a figure - placeholder for Gradio compatibility"""
    return fig_to_html(fig)

def get_table_download_link(df, filename, text):
    """Generate an HTML representation of a dataframe - placeholder for Gradio compatibility"""
    return df_to_html_table(df)

def format_pos_token(token, pos, explanation=""):
    """Format a token with its part-of-speech tag in HTML"""
    # Define colors for different POS types
    pos_colors = {
        'NOUN': '#E3F2FD',  # Light blue
        'PROPN': '#E3F2FD',  # Light blue (same as NOUN)
        'VERB': '#E8F5E9',  # Light green
        'ADJ': '#FFF8E1',   # Light yellow
        'ADV': '#F3E5F5',   # Light purple
        'ADP': '#EFEBE9',   # Light brown
        'PRON': '#E8EAF6',  # Light indigo
        'DET': '#E0F7FA',   # Light cyan
        'CONJ': '#FBE9E7',  # Light deep orange
        'CCONJ': '#FBE9E7',  # Light deep orange (for compatibility)
        'SCONJ': '#FBE9E7',  # Light deep orange (for compatibility)
        'NUM': '#FFEBEE',   # Light red
        'PART': '#F1F8E9',  # Light light green
        'INTJ': '#FFF3E0',  # Light orange
        'PUNCT': '#FAFAFA',  # Light grey
        'SYM': '#FAFAFA',   # Light grey (same as PUNCT)
        'X': '#FAFAFA',     # Light grey (for other)
    }
    
    # Get color for this POS tag, default to light grey if not found
    bg_color = pos_colors.get(pos, '#FAFAFA')
    
    # Create HTML for the token with tooltip
    if explanation:
        return f'<span class="pos-token" style="background-color: {bg_color}; border: 1px solid #ccc; padding: 3px 6px; margin: 2px; display: inline-block; border-radius: 3px;" title="{explanation}">{token} <small style="color: #666; font-size: 0.8em;">({pos})</small></span>'
    else:
        return f'<span class="pos-token" style="background-color: {bg_color}; border: 1px solid #ccc; padding: 3px 6px; margin: 2px; display: inline-block; border-radius: 3px;">{token} <small style="color: #666; font-size: 0.8em;">({pos})</small></span>'

def create_entity_span(text, entity_type, explanation=""):
    """Format a named entity with its type in HTML"""
    # Define colors for different entity types
    entity_colors = {
        'PERSON': '#E3F2FD',      # Light blue
        'ORG': '#E8F5E9',         # Light green
        'GPE': '#FFF8E1',         # Light yellow
        'LOC': '#F3E5F5',         # Light purple
        'PRODUCT': '#EFEBE9',     # Light brown
        'EVENT': '#E8EAF6',       # Light indigo
        'WORK_OF_ART': '#E0F7FA', # Light cyan
        'LAW': '#FBE9E7',         # Light deep orange
        'LANGUAGE': '#FFEBEE',    # Light red
        'DATE': '#F1F8E9',        # Light light green
        'TIME': '#FFF3E0',        # Light orange
        'PERCENT': '#FAFAFA',     # Light grey
        'MONEY': '#FAFAFA',       # Light grey
        'QUANTITY': '#FAFAFA',    # Light grey
        'ORDINAL': '#FAFAFA',     # Light grey
        'CARDINAL': '#FAFAFA',    # Light grey
    }
    
    # Get color for this entity type, default to light grey if not found
    bg_color = entity_colors.get(entity_type, '#FAFAFA')
    
    # Create HTML for the entity with tooltip
    if explanation:
        return f'<span class="entity-token" style="background-color: {bg_color}; border: 1px solid #ccc; padding: 3px 6px; margin: 2px; display: inline-block; border-radius: 3px;" title="{explanation}">{text} <small style="color: #666; font-size: 0.8em;">({entity_type})</small></span>'
    else:
        return f'<span class="entity-token" style="background-color: {bg_color}; border: 1px solid #ccc; padding: 3px 6px; margin: 2px; display: inline-block; border-radius: 3px;">{text} <small style="color: #666; font-size: 0.8em;">({entity_type})</small></span>'

def create_sentiment_color(score):
    """Create color based on sentiment score"""
    if score > 0.1:
        return '#4CAF50'  # Green for positive
    elif score < -0.1:
        return '#F44336'  # Red for negative
    else:
        return '#FF9800'  # Orange for neutral

def format_sentiment_score(score, label):
    """Format sentiment score with appropriate color"""
    color = create_sentiment_color(score)
    return f'<span style="color: {color}; font-weight: bold;">{label} ({score:.3f})</span>'

def create_progress_bar(value, max_value=1.0, color='#1976D2'):
    """Create HTML progress bar"""
    percentage = (value / max_value) * 100
    return f'''
    <div class="progress mb-2" style="height: 20px;">
        <div class="progress-bar" role="progressbar" style="width: {percentage}%; background-color: {color};" 
             aria-valuenow="{value}" aria-valuemin="0" aria-valuemax="{max_value}">
            {value:.3f}
        </div>
    </div>
    '''

def create_confidence_gauge(score, label):
    """Create confidence gauge visualization"""
    color = '#4CAF50' if score > 0.7 else '#FF9800' if score > 0.4 else '#F44336'
    return f'''
    <div class="text-center">
        <div class="display-6 text-{color.replace('#', '')}" style="color: {color};">
            {score:.1%}
        </div>
        <div class="small text-muted">{label}</div>
    </div>
    '''
