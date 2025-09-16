import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import time

from utils.model_loader import load_translator
from utils.helpers import fig_to_html, df_to_html_table

def translation_handler(text_input, source_lang="auto", target_lang="en"):
    """Show machine translation capabilities."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Machine Translation</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-language"></i>
    Machine translation converts text from one language to another while preserving meaning and context as accurately as possible.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Model Used:</h4>
        <ul>
            <li><b>Helsinki-NLP/opus-mt</b> - A collection of pre-trained neural machine translation models</li>
            <li><b>Capabilities</b> - Translates between various language pairs with good accuracy</li>
            <li><b>Architecture</b> - Transformer-based sequence-to-sequence model</li>
        </ul>
    </div>
    """)
    
    try:
        # Check if text is empty
        if not text_input.strip():
            output_html.append("""
            <div class="alert alert-warning">
                <h3>No Text Provided</h3>
                <p>Please enter some text to translate.</p>
            </div>
            """)
            output_html.append('</div>')  # Close result-area div
            return '\n'.join(output_html)
        
        # Display source text
        output_html.append('<h3 class="task-subheader">Source Text</h3>')
        
        # Language mapping for display
        language_names = {
            "auto": "Auto-detect",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
            "ja": "Japanese",
            "pt": "Portuguese",
            "it": "Italian"
        }
        
        source_lang_display = language_names.get(source_lang, source_lang)
        target_lang_display = language_names.get(target_lang, target_lang)
        
        # Format source text info
        output_html.append(f"""
        <div class="mb-2">
            <span class="badge bg-primary">
                {source_lang_display}
            </span>
        </div>
        """)
        
        # Display source text
        output_html.append(f'<div class="card"><div class="card-body">{text_input}</div></div>')
        
        # Load translation model
        translator = load_translator(source_lang, target_lang)
        
        # Translate text
        start_time = time.time()
        
        # Check text length and apply limit if needed
        MAX_TEXT_LENGTH = 500  # Characters
        truncated = False
        
        if len(text_input) > MAX_TEXT_LENGTH:
            truncated_text = text_input[:MAX_TEXT_LENGTH]
            truncated = True
        else:
            truncated_text = text_input
        
        # Perform translation
        translation = translator(truncated_text)
        translated_text = translation[0]['translation_text']
        
        # Calculate processing time
        translation_time = time.time() - start_time
        
        # Display translation results
        output_html.append('<h3 class="task-subheader">Translation</h3>')
        
        # Show target language
        output_html.append(f"""
        <div class="mb-2">
            <span class="badge bg-success">
                {target_lang_display}
            </span>
        </div>
        """)
        
        # Display translated text
        output_html.append(f'<div class="card"><div class="card-body bg-light">{translated_text}</div></div>')
        
        # Show truncation warning if needed
        if truncated:
            output_html.append(f"""
            <div class="alert alert-warning">
                <p class="mb-0"><b>⚠️ Note:</b> Your text was truncated to {MAX_TEXT_LENGTH} characters due to model limitations. Only the first part was translated.</p>
            </div>
            """)
        
        # Translation statistics
        output_html.append('<h3 class="task-subheader">Translation Analysis</h3>')
        
        # Calculate basic stats
        source_chars = len(text_input)
        source_words = len(text_input.split())
        target_chars = len(translated_text)
        target_words = len(translated_text.split())
        
        # Display stats in a nice format
        output_html.append(f"""
        <div class="row text-center mb-4">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <div class="display-4 text-primary">{source_words}</div>
                        <div>Source Words</div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <div class="display-4 text-success">{target_words}</div>
                        <div>Translated Words</div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <div class="display-4 text-warning">{translation_time:.2f}s</div>
                        <div>Processing Time</div>
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # Length comparison
        output_html.append('<h4>Length Comparison</h4>')
        
        # Create bar chart comparing text lengths
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create grouped bar chart
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, [source_words, source_chars], width, label='Source Text', color='#1976D2')
        ax.bar(x + width/2, [target_words, target_chars], width, label='Translated Text', color='#4CAF50')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Word Count', 'Character Count'])
        ax.legend()
        
        # Add value labels on top of bars
        for i, v in enumerate([source_words, source_chars]):
            ax.text(i - width/2, v + 0.5, str(v), ha='center')
        
        for i, v in enumerate([target_words, target_chars]):
            ax.text(i + width/2, v + 0.5, str(v), ha='center')
        
        plt.title('Source vs. Translation Length Comparison')
        plt.tight_layout()
        
        output_html.append(fig_to_html(fig))
        
        # Expansion/contraction ratio
        word_ratio = target_words / source_words if source_words > 0 else 0
        char_ratio = target_chars / source_chars if source_chars > 0 else 0
        
        expansion_type = "expansion" if word_ratio > 1.1 else "contraction" if word_ratio < 0.9 else "similar length"
        
        output_html.append(f"""
        <div class="alert alert-info">
            <h4>Translation Length Analysis</h4>
            <p>The translation shows <b>{expansion_type}</b> compared to the source text.</p>
            <ul>
                <li>Word ratio: {word_ratio:.2f} (target/source)</li>
                <li>Character ratio: {char_ratio:.2f} (target/source)</li>
            </ul>
            <p><small>Note: Different languages naturally have different word and character counts when expressing the same meaning.</small></p>
        </div>
        """)
        
        # Language characteristics comparison
        source_avg_word_len = source_chars / source_words if source_words > 0 else 0
        target_avg_word_len = target_chars / target_words if target_words > 0 else 0
        
        output_html.append('<h4>Language Characteristics</h4>')
        
        # Create comparison table
        lang_data = {
            'Metric': ['Average Word Length', 'Words per Character', 'Characters per Word'],
            f'Source ({source_lang_display})': [
                f"{source_avg_word_len:.2f} chars",
                f"{source_words / source_chars:.3f}" if source_chars > 0 else "N/A",
                f"{source_chars / source_words:.2f}" if source_words > 0 else "N/A"
            ],
            f'Target ({target_lang_display})': [
                f"{target_avg_word_len:.2f} chars",
                f"{target_words / target_chars:.3f}" if target_chars > 0 else "N/A",
                f"{target_chars / target_words:.2f}" if target_words > 0 else "N/A"
            ]
        }
        
        lang_df = pd.DataFrame(lang_data)
        
        output_html.append(df_to_html_table(lang_df))
        
        # Alternative translations section
        output_html.append('<h3 class="task-subheader">Alternative Translation Options</h3>')
        output_html.append('<p>Machine translation models often have different ways of translating the same text. Here are some general tips for better translations:</p>')
        
        output_html.append("""
        <div class="alert alert-info">
            <h4>Tips for Better Machine Translation</h4>
            <ul class="mb-0">
                <li><b>Use clear, simple language</b> in your source text</li>
                <li><b>Avoid idioms and slang</b> that may not translate well across cultures</li>
                <li><b>Break up long, complex sentences</b> into simpler ones</li>
                <li><b>Provide context</b> when dealing with ambiguous terms</li>
                <li><b>Review and post-edit</b> machine translations for important documents</li>
            </ul>
        </div>
        """)
        
        # Common translation challenges
        output_html.append('<h4>Common Translation Challenges</h4>')
        
        challenge_data = {
            'Challenge': [
                'Ambiguity', 
                'Idioms & Expressions', 
                'Cultural References', 
                'Technical Terminology',
                'Grammatical Differences'
            ],
            'Description': [
                'Words with multiple meanings may be incorrectly translated without proper context',
                'Expressions that are unique to a culture often lose meaning when translated literally',
                'References to culture-specific concepts may not have direct equivalents',
                'Specialized terminology may not translate accurately without domain-specific models',
                'Different languages have different grammatical structures that can affect translation'
            ],
            'Example': [
                '"Bank" could mean financial institution or river edge',
                '"It\'s raining cats and dogs" translated literally loses its meaning',
                'References to local holidays or customs may be confusing when translated',
                'Medical or legal terms often need specialized translation knowledge',
                'Languages differ in word order, gender agreement, verb tenses, etc.'
            ]
        }
        
        challenge_df = pd.DataFrame(challenge_data)
        
        output_html.append(df_to_html_table(challenge_df))
        
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h3>Translation Error</h3>
            <p>{str(e)}</p>
            <p>This could be due to an unsupported language pair or an issue loading the translation model.</p>
        </div>
        """)
    
    # About Machine Translation section
    output_html.append("""
    <div class="card mt-4">
        <div class="card-header">
            <h4 class="mb-0">
                <i class="fas fa-info-circle"></i>
                About Machine Translation
            </h4>
        </div>
        <div class="card-body">
            <h5>What is Machine Translation?</h5>
            
            <p>Machine translation is the automated translation of text from one language to another using computer software.
            Modern machine translation systems use neural networks to understand and generate text, leading to significant
            improvements in fluency and accuracy compared to older rule-based or statistical systems.</p>
            
            <h5>Types of Machine Translation:</h5>
            
            <ul>
                <li><b>Rule-based MT</b> - Uses linguistic rules crafted by human experts</li>
                <li><b>Statistical MT</b> - Uses statistical models trained on parallel texts</li>
                <li><b>Neural MT</b> - Uses deep learning and neural networks (current state-of-the-art)</li>
                <li><b>Hybrid MT</b> - Combines multiple approaches for better results</li>
            </ul>
            
            <h5>Applications:</h5>
            
            <ul>
                <li><b>Website localization</b> - Translating web content for international audiences</li>
                <li><b>Document translation</b> - Quickly obtaining translations of documents</li>
                <li><b>Real-time communication</b> - Enabling conversations across language barriers</li>
                <li><b>E-commerce</b> - Making product listings available in multiple languages</li>
                <li><b>Content accessibility</b> - Making information available to speakers of different languages</li>
            </ul>
        </div>
    </div>
    """)
    
    output_html.append('</div>')  # Close result-area div
    
    return '\n'.join(output_html)
