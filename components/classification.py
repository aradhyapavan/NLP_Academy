import matplotlib.pyplot as plt
import pandas as pd
from utils.model_loader import load_zero_shot
from utils.helpers import fig_to_html, df_to_html_table

def classification_handler(text_input, scenario="Sentiment", multi_label=False, custom_labels=""):
    """Show zero-shot classification capabilities."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Zero-shot Classification</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-tags"></i>
    Zero-shot classification can categorize text into arbitrary classes without having been specifically trained on those categories.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Model Used:</h4>
        <ul>
            <li><b>facebook/bart-large-mnli</b> - BART model fine-tuned on MultiNLI dataset</li>
            <li><b>Capabilities</b> - Can classify text into any user-defined categories</li>
            <li><b>Performance</b> - Best performance on distinct, well-defined categories</li>
        </ul>
    </div>
    """)
    
    # Classification scenarios
    scenarios = {
        "Sentiment": ["positive", "negative", "neutral"],
        "Emotion": ["joy", "sadness", "anger", "fear", "surprise"],
        "Writing Style": ["formal", "informal", "technical", "creative", "persuasive"],
        "Intent": ["inform", "persuade", "entertain", "instruct"],
        "Content Type": ["news", "opinion", "review", "instruction", "narrative"],
        "Audience Level": ["beginner", "intermediate", "advanced", "expert"],
        "Custom": []
    }
    
    try:
        # Get labels based on scenario
        if scenario == "Custom":
            labels = [label.strip() for label in custom_labels.split("\n") if label.strip()]
            if not labels:
                output_html.append("""
                <div class="alert alert-warning">
                    <h3>No Custom Categories</h3>
                    <p>Please enter at least one custom category.</p>
                </div>
                """)
                output_html.append('</div>')  # Close result-area div
                return '\n'.join(output_html)
        else:
            labels = scenarios[scenario]
        
        # Update multi-label default for certain categories
        if scenario in ["Emotion", "Intent", "Content Type"] and not multi_label:
            multi_label = True
        
        # Load model
        classifier = load_zero_shot()
        
        # Classification process
        result = classifier(text_input, labels, multi_label=multi_label)
        
        # Display results
        output_html.append('<h3 class="task-subheader">Classification Results</h3>')
        
        # Create DataFrame
        class_df = pd.DataFrame({
            'Category': result['labels'],
            'Confidence': result['scores']
        })
        
        # Visualization
        fig = plt.figure(figsize=(10, 6))
        bars = plt.barh(class_df['Category'], class_df['Confidence'], color='#1976D2')
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{bar.get_width():.1%}", va='center')
        
        plt.xlim(0, 1.1)
        plt.xlabel('Confidence Score')
        plt.title(f'{scenario} Classification')
        plt.tight_layout()
        
        # Layout with vertical stacking - Chart first
        output_html.append('<div class="row mb-4">')
        output_html.append('<div class="col-12">')
        output_html.append('<h4>Classification Confidence Chart</h4>')
        output_html.append(fig_to_html(fig))
        output_html.append('</div>')
        output_html.append('</div>')  # Close chart row
        
        # Data table and result in next row
        output_html.append('<div class="row">')
        output_html.append('<div class="col-md-6">')
        output_html.append('<h4>Detailed Results</h4>')
        output_html.append(df_to_html_table(class_df))
        output_html.append('</div>')
        
        # Top result
        output_html.append('<div class="col-md-6">')
        top_class = class_df.iloc[0]['Category']
        top_score = class_df.iloc[0]['Confidence']
        
        output_html.append(f"""
        <div class="alert alert-primary">
            <h3>Primary Classification</h3>
            <p class="h4">{top_class}</p>
            <p>Confidence: {top_score:.1%}</p>
        </div>
        """)
        
        output_html.append('</div>')  # Close result column
        output_html.append('</div>')  # Close row
        
        # Multiple categories (if multi-label)
        if multi_label:
            # Get all categories with significant confidence
            significant_classes = class_df[class_df['Confidence'] > 0.5]
            
            if len(significant_classes) > 1:
                output_html.append(f"""
                <div class="alert alert-info">
                    <h3>Multiple Categories Detected</h3>
                    <p>This text appears to belong to multiple categories:</p>
                </div>
                """)
                
                category_list = []
                for _, row in significant_classes.iterrows():
                    category_list.append(f"<li><b>{row['Category']}</b> ({row['Confidence']:.1%})</li>")
                
                output_html.append(f"<ul>{''.join(category_list)}</ul>")
    
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h3>Error</h3>
            <p>Failed to classify text: {str(e)}</p>
        </div>
        """)
    
    # About zero-shot classification
    output_html.append("""
    <div class="card mt-4">
        <div class="card-header">
            <h4 class="mb-0">
                <i class="fas fa-info-circle"></i>
                About Zero-shot Classification
            </h4>
        </div>
        <div class="card-body">
            <h5>What is Zero-shot Classification?</h5>
            
            <p>Unlike traditional classifiers that need to be trained on examples from each category, 
            zero-shot classification can categorize text into arbitrary classes it has never seen 
            during training.</p>
            
            <h5>How it works:</h5>
            
            <ol>
                <li>The model converts your text and each potential category into embeddings</li>
                <li>It calculates how likely the text entails or belongs to each category</li>
                <li>The model ranks categories by confidence scores</li>
            </ol>
            
            <h5>Benefits:</h5>
            
            <ul>
                <li>Flexibility to classify into any categories without retraining</li>
                <li>Can work with domain-specific or custom categories</li>
                <li>Useful for exploratory analysis or when training data is limited</li>
            </ul>
        </div>
    </div>
    """)
    
    output_html.append('</div>')  # Close result-area div
    
    return '\n'.join(output_html)
