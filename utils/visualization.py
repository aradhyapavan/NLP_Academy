import matplotlib.pyplot as plt
import seaborn as sns

def apply_custom_css():
    """Load custom CSS for the Flask interface"""
    css_file_path = "static/css/style.css"
    try:
        with open(css_file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load custom CSS: {e}")
        return ""

def setup_mpl_style():
    """Setup matplotlib style for consistent visualizations"""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_style("whitegrid")
    except:
        # Fallback if seaborn style is not available
        plt.style.use('default')
    
    # Configure matplotlib for better visuals
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def create_bar_chart(labels, values, title, xlabel, ylabel, color='#1976D2'):
    """Create a matplotlib bar chart"""
    setup_mpl_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=color)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def create_horizontal_bar_chart(labels, values, title, xlabel, ylabel, color='#1976D2'):
    """Create a matplotlib horizontal bar chart"""
    setup_mpl_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color=color)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()  # To have the highest value at the top
    plt.tight_layout()
    
    return fig

def create_pie_chart(labels, values, title, colors=None):
    """Create a matplotlib pie chart"""
    setup_mpl_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if colors is None:
        colors = ['#1976D2', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#FFC107', '#795548']
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
    ax.set_title(title)
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    return fig

def create_line_chart(x_values, y_values, title, xlabel, ylabel, color='#1976D2'):
    """Create a matplotlib line chart"""
    setup_mpl_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, y_values, color=color, linewidth=2, marker='o')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def create_scatter_plot(x_values, y_values, title, xlabel, ylabel, color='#1976D2'):
    """Create a matplotlib scatter plot"""
    setup_mpl_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_values, y_values, color=color, alpha=0.6, s=50)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def create_heatmap(data, title, xlabel, ylabel, cmap='YlGnBu'):
    """Create a matplotlib heatmap"""
    setup_mpl_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    return fig

def create_word_cloud_placeholder(text, title="Word Cloud"):
    """Create a placeholder for word cloud visualization"""
    setup_mpl_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a simple text visualization as placeholder
    ax.text(0.5, 0.5, f"Word Cloud: {title}\n\n{text[:100]}...", 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def create_network_graph(edges, nodes, title="Network Graph"):
    """Create a network graph visualization"""
    setup_mpl_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Simple network visualization
    if edges and nodes:
        # Extract node positions (simplified)
        pos = {}
        for i, node in enumerate(nodes):
            angle = 2 * 3.14159 * i / len(nodes)
            pos[node] = (0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle))
        
        # Draw edges
        for edge in edges:
            if len(edge) >= 2:
                x1, y1 = pos.get(edge[0], (0, 0))
                x2, y2 = pos.get(edge[1], (0, 0))
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=1)
        
        # Draw nodes
        for node, (x, y) in pos.items():
            ax.scatter(x, y, s=200, c='lightblue', edgecolors='black', linewidth=2)
            ax.text(x, y, str(node), ha='center', va='center', fontsize=8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def create_gauge_chart(value, max_value=1.0, title="Gauge Chart"):
    """Create a gauge chart visualization"""
    setup_mpl_style()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # Color based on value
    if value / max_value > 0.7:
        color = '#4CAF50'  # Green
    elif value / max_value > 0.4:
        color = '#FF9800'  # Orange
    else:
        color = '#F44336'  # Red
    
    ax.fill_between(theta, 0, r, alpha=0.3, color=color)
    ax.plot(theta, r, color=color, linewidth=3)
    
    # Add value indicator
    indicator_theta = np.pi * (1 - value / max_value)
    ax.plot([indicator_theta, indicator_theta], [0, 1], color='black', linewidth=4)
    
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add value text
    ax.text(0, 0, f'{value:.2f}', ha='center', va='center', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_comparison_chart(categories, values1, values2, title, xlabel, ylabel, 
                          label1="Series 1", label2="Series 2", color1='#1976D2', color2='#4CAF50'):
    """Create a comparison bar chart"""
    setup_mpl_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, values1, width, label=label1, color=color1)
    bars2 = ax.bar(x + width/2, values2, width, label=label2, color=color2)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    plt.tight_layout()
    return fig
