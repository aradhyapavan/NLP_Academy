# Utils package for NLP Ultimate Tutorial Flask Application

from .helpers import (
    fig_to_html,
    df_to_html_table,
    text_statistics,
    create_text_length_chart,
    format_pos_token,
    create_entity_span,
    create_sentiment_color,
    format_sentiment_score,
    create_progress_bar,
    create_confidence_gauge
)

from .model_loader import (
    download_nltk_resources,
    load_spacy,
    load_sentiment_analyzer,
    load_emotion_classifier,
    load_summarizer,
    load_qa_pipeline,
    load_translator,
    load_text_generator,
    load_zero_shot,
    load_embedding_model,
    initialize_all_models,
    get_model_status,
    clear_models
)

from .visualization import (
    setup_mpl_style,
    create_bar_chart,
    create_horizontal_bar_chart,
    create_pie_chart,
    create_line_chart,
    create_scatter_plot,
    create_heatmap,
    create_word_cloud_placeholder,
    create_network_graph,
    create_gauge_chart,
    create_comparison_chart
)

__all__ = [
    # Helpers
    'fig_to_html',
    'df_to_html_table',
    'text_statistics',
    'create_text_length_chart',
    'format_pos_token',
    'create_entity_span',
    'create_sentiment_color',
    'format_sentiment_score',
    'create_progress_bar',
    'create_confidence_gauge',
    
    # Model Loader
    'download_nltk_resources',
    'load_spacy',
    'load_sentiment_analyzer',
    'load_emotion_classifier',
    'load_summarizer',
    'load_qa_pipeline',
    'load_translator',
    'load_text_generator',
    'load_zero_shot',
    'load_embedding_model',
    'initialize_all_models',
    'get_model_status',
    'clear_models',
    
    # Visualization
    'setup_mpl_style',
    'create_bar_chart',
    'create_horizontal_bar_chart',
    'create_pie_chart',
    'create_line_chart',
    'create_scatter_plot',
    'create_heatmap',
    'create_word_cloud_placeholder',
    'create_network_graph',
    'create_gauge_chart',
    'create_comparison_chart'
]
