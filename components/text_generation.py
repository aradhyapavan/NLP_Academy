import matplotlib.pyplot as plt
import pandas as pd
import nltk
import time

from utils.model_loader import load_text_generator
from utils.helpers import fig_to_html, df_to_html_table

def text_generation_handler(text_input, max_length=100, temperature=0.7, top_p=0.9, num_sequences=1):
    """Show text generation capabilities."""
    output_html = []
    
    # Add result area container
    output_html.append('<div class="result-area">')
    output_html.append('<h2 class="task-header">Text Generation</h2>')
    
    output_html.append("""
    <div class="alert alert-info">
    <i class="fas fa-info-circle"></i>
    Text generation models can continue or expand on a given text prompt, creating new content that follows the style and context of the input.
    </div>
    """)
    
    # Model info
    output_html.append("""
    <div class="alert alert-info">
        <h4><i class="fas fa-tools"></i> Model Used:</h4>
        <ul>
            <li><b>GPT-2</b> - 124M parameter language model trained on a diverse corpus of internet text</li>
            <li><b>Capabilities</b> - Can generate coherent text continuations and completions</li>
            <li><b>Limitations</b> - May occasionally produce repetitive or nonsensical content</li>
        </ul>
    </div>
    """)
    
    try:
        # Check text length and possibly truncate
        MAX_PROMPT_LENGTH = 100  # tokens
        
        # Count tokens (rough approximation)
        token_count = len(text_input.split())
        
        # Truncate if necessary
        if token_count > MAX_PROMPT_LENGTH:
            prompt_text = " ".join(text_input.split()[:MAX_PROMPT_LENGTH])
            output_html.append("""
            <div class="alert alert-warning">
                <p class="mb-0">⚠️ Text truncated to approximately 100 tokens for better generation results.</p>
            </div>
            """)
        else:
            prompt_text = text_input
        
        # Display prompt
        output_html.append('<h3 class="task-subheader">Prompt</h3>')
        output_html.append(f'<div class="card"><div class="card-body">{prompt_text}</div></div>')
        
        # Load model
        text_generator = load_text_generator()
        
        # Set up generation parameters
        generation_kwargs = {
            "max_length": token_count + max_length,
            "num_return_sequences": num_sequences,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "no_repeat_ngram_size": 2,
            "pad_token_id": 50256  # GPT-2's pad token ID
        }
        
        # Generate text
        start_time = time.time()
        result = text_generator(prompt_text, **generation_kwargs)
        generation_time = time.time() - start_time
        
        # Display results
        output_html.append('<h3 class="task-subheader">Generated Text</h3>')
        
        for i, sequence in enumerate(result):
            generated_text = sequence['generated_text']
            new_text = generated_text[len(prompt_text):]
            
            # Display in a nice format with the prompt and generated text distinguished
            if num_sequences > 1:
                output_html.append(f'<h4>Version {i+1}</h4>')
            
            output_html.append(f"""
            <div class="card">
                <div class="card-body">
                    <span class="text-muted">{prompt_text}</span>
                    <span class="text-primary fw-bold">{new_text}</span>
                </div>
            </div>
            """)
            
            # Generation stats for this sequence
            prompt_tokens = len(prompt_text.split())
            gen_tokens = len(new_text.split())
            
            # Calculate average word length as a crude complexity metric
            avg_word_len = sum(len(word) for word in new_text.split()) / max(1, len(new_text.split()))
            
            output_html.append(f"""
            <div class="alert alert-success">
                <h4 class="mb-3">Generation Statistics</h4>
                <div class="row">
                    <div class="col-md-6">
                        <p><b>Prompt length:</b> {prompt_tokens} tokens</p>
                        <p><b>Generated length:</b> {gen_tokens} tokens</p>
                        <p><b>Total length:</b> {prompt_tokens + gen_tokens} tokens</p>
                    </div>
                    <div class="col-md-6">
                        <p><b>Temperature:</b> {temperature}</p>
                        <p><b>Top-p:</b> {top_p}</p>
                        <p><b>Avg word length:</b> {avg_word_len:.2f} characters</p>
                    </div>
                </div>
                <p><b>Generation time:</b> {generation_time:.2f} seconds</p>
            </div>
            """)
            
            # Option to see full text
            output_html.append(f"""
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">
                        <button class="btn btn-link" type="button" data-bs-toggle="collapse" data-bs-target="#fullText{i}" aria-expanded="false">
                            Show full text (copy-paste friendly)
                        </button>
                    </h5>
                </div>
                <div class="collapse" id="fullText{i}">
                    <div class="card-body">
                        <div class="text-content" style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word; max-height: 500px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; line-height: 1.6;">{generated_text}</div>
                    </div>
                </div>
            </div>
            """)
        
        # Generate a text complexity analysis
        if len(result) > 0:
            output_html.append('<h3 class="task-subheader">Text Analysis</h3>')
            
            # Get the first generated text for analysis
            full_text = result[0]['generated_text']
            prompt_words = prompt_text.split()
            full_words = full_text.split()
            generated_words = full_words[len(prompt_words):]
            
            # Analyze word length distribution
            prompt_word_lengths = [len(word) for word in prompt_words]
            generated_word_lengths = [len(word) for word in generated_words]
            
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot histograms
            bins = range(1, 16)  # Word lengths from 1 to 15
            ax.hist(prompt_word_lengths, bins=bins, alpha=0.7, label='Prompt', color='#1976D2')
            ax.hist(generated_word_lengths, bins=bins, alpha=0.7, label='Generated', color='#4CAF50')
            
            ax.set_xlabel('Word Length (characters)')
            ax.set_ylabel('Frequency')
            ax.set_title('Word Length Distribution: Prompt vs Generated')
            ax.legend()
            ax.grid(alpha=0.3)
            
            output_html.append(fig_to_html(fig))
            
            # Calculate some linguistic statistics
            prompt_avg_word_len = sum(prompt_word_lengths) / len(prompt_word_lengths) if prompt_word_lengths else 0
            generated_avg_word_len = sum(generated_word_lengths) / len(generated_word_lengths) if generated_word_lengths else 0
            
            # Create comparison table
            stats_data = {
                'Metric': ['Word count', 'Average word length', 'Unique words', 'Lexical diversity*'],
                'Prompt': [
                    len(prompt_words),
                    f"{prompt_avg_word_len:.2f}",
                    len(set(word.lower() for word in prompt_words)),
                    f"{len(set(word.lower() for word in prompt_words)) / len(prompt_words):.2f}" if prompt_words else "0"
                ],
                'Generated': [
                    len(generated_words),
                    f"{generated_avg_word_len:.2f}",
                    len(set(word.lower() for word in generated_words)),
                    f"{len(set(word.lower() for word in generated_words)) / len(generated_words):.2f}" if generated_words else "0"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            
            output_html.append('<div class="mt-3">')
            output_html.append(df_to_html_table(stats_df))
            output_html.append('<p><small>*Lexical diversity = unique words / total words</small></p>')
            output_html.append('</div>')
            
            # Show tips for better results
            output_html.append("""
            <div class="alert alert-info">
                <h4>Tips for Better Generation Results</h4>
                <ul class="mb-0">
                    <li><b>Be specific</b> - More detailed prompts give the model better context</li>
                    <li><b>Format matters</b> - If you want a list, start with a list item; if you want dialogue, include dialogue format</li>
                    <li><b>Play with temperature</b> - Lower values (0.3-0.5) for focused, consistent text; higher values (0.7-1.0) for creative, varied output</li>
                    <li><b>Try multiple generations</b> - Generate several options to pick the best result</li>
                </ul>
            </div>
            """)
    
    except Exception as e:
        output_html.append(f"""
        <div class="alert alert-danger">
            <h3>Error</h3>
            <p>Failed to generate text: {str(e)}</p>
        </div>
        """)
    
    # About Text Generation section
    output_html.append("""
    <div class="card mt-4">
        <div class="card-header">
            <h4 class="mb-0">
                <i class="fas fa-info-circle"></i>
                About Text Generation
            </h4>
        </div>
        <div class="card-body">
            <h5>What is Text Generation?</h5>
            
            <p>Text generation is the task of creating human-like text using machine learning models. Modern text generation
            systems use large neural networks trained on vast amounts of text data to predict the next tokens in a sequence.</p>
            
            <h5>How It Works:</h5>
            
            <ol>
                <li><b>Training</b> - Models learn patterns in language by predicting the next word in billions of text examples</li>
                <li><b>Prompting</b> - You provide a starting text that gives context and direction</li>
                <li><b>Generation</b> - The model repeatedly predicts the most likely next token based on previous context</li>
                <li><b>Sampling</b> - Various techniques (temperature, top-p) control the randomness and creativity of output</li>
            </ol>
            
            <h5>Applications:</h5>
            
            <ul>
                <li><b>Content creation</b> - Drafting articles, stories, and marketing copy</li>
                <li><b>Assistive writing</b> - Helping with email drafting, summarization, and editing</li>
                <li><b>Conversational AI</b> - Powering chatbots and digital assistants</li>
                <li><b>Code generation</b> - Assisting developers with coding tasks</li>
                <li><b>Creative writing</b> - Generating stories, poetry, and other creative content</li>
            </ul>
        </div>
    </div>
    """)
    
    output_html.append('</div>')  # Close result-area div
    
    return '\n'.join(output_html)
