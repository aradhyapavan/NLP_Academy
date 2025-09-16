// Component-specific JavaScript for NLP Ultimate Tutorial

// POS Tagging functionality
class POSTagging {
    static highlightTokens(tokens, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = tokens.map(token => {
            const color = this.getPOSColor(token.pos);
            return `<span class="pos-token" style="background-color: ${color};" 
                     title="${token.explanation || ''}">${token.text} 
                     <small>(${token.pos})</small></span>`;
        }).join(' ');
    }
    
    static getPOSColor(pos) {
        const colors = {
            'NOUN': '#E3F2FD', 'PROPN': '#E3F2FD', 'VERB': '#E8F5E9',
            'ADJ': '#FFF8E1', 'ADV': '#F3E5F5', 'ADP': '#EFEBE9',
            'PRON': '#E8EAF6', 'DET': '#E0F7FA', 'CONJ': '#FBE9E7',
            'NUM': '#FFEBEE', 'PART': '#F1F8E9', 'INTJ': '#FFF3E0',
            'PUNCT': '#FAFAFA', 'SYM': '#FAFAFA', 'X': '#FAFAFA'
        };
        return colors[pos] || '#FAFAFA';
    }
}

// Named Entity Recognition functionality
class NER {
    static highlightEntities(entities, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = entities.map(entity => {
            const color = this.getEntityColor(entity.type);
            return `<span class="entity-token" style="background-color: ${color};" 
                     title="${entity.explanation || ''}">${entity.text} 
                     <small>(${entity.type})</small></span>`;
        }).join(' ');
    }
    
    static getEntityColor(type) {
        const colors = {
            'PERSON': '#E3F2FD', 'ORG': '#E8F5E9', 'GPE': '#FFF8E1',
            'LOC': '#F3E5F5', 'PRODUCT': '#EFEBE9', 'EVENT': '#E8EAF6',
            'WORK_OF_ART': '#E0F7FA', 'LAW': '#FBE9E7', 'LANGUAGE': '#FFEBEE',
            'DATE': '#F1F8E9', 'TIME': '#FFF3E0', 'PERCENT': '#FAFAFA',
            'MONEY': '#FAFAFA', 'QUANTITY': '#FAFAFA', 'ORDINAL': '#FAFAFA',
            'CARDINAL': '#FAFAFA'
        };
        return colors[type] || '#FAFAFA';
    }
}

// Sentiment Analysis functionality
class SentimentAnalysis {
    static createGauge(score, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const color = this.getSentimentColor(score);
        const label = this.getSentimentLabel(score);
        
        container.innerHTML = `
            <div class="sentiment-gauge">
                <div class="sentiment-score" style="color: ${color};">${score.toFixed(3)}</div>
                <div class="sentiment-label" style="color: ${color};">${label}</div>
            </div>
        `;
    }
    
    static getSentimentColor(score) {
        if (score > 0.1) return '#4CAF50';
        if (score < -0.1) return '#F44336';
        return '#FF9800';
    }
    
    static getSentimentLabel(score) {
        if (score > 0.1) return 'Positive';
        if (score < -0.1) return 'Negative';
        return 'Neutral';
    }
}

// Text Generation functionality
class TextGeneration {
    static displayGeneratedText(prompt, generated, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = `
            <div class="generated-text">
                <span class="prompt-text">${prompt}</span>
                <span class="generated-content">${generated}</span>
            </div>
        `;
    }
    
    static createParameterControls(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <label for="temperature" class="form-label">Temperature</label>
                    <input type="range" class="form-range" id="temperature" min="0.1" max="1.5" value="0.7" step="0.1">
                    <div class="d-flex justify-content-between">
                        <small>0.1</small>
                        <small id="temperature-value">0.7</small>
                        <small>1.5</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <label for="top-p" class="form-label">Top-p</label>
                    <input type="range" class="form-range" id="top-p" min="0.1" max="1.0" value="0.9" step="0.1">
                    <div class="d-flex justify-content-between">
                        <small>0.1</small>
                        <small id="top-p-value">0.9</small>
                        <small>1.0</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <label for="max-length" class="form-label">Max Length</label>
                    <input type="range" class="form-range" id="max-length" min="30" max="250" value="100" step="10">
                    <div class="d-flex justify-content-between">
                        <small>30</small>
                        <small id="max-length-value">100</small>
                        <small>250</small>
                    </div>
                </div>
            </div>
        `;
        
        // Add event listeners for parameter updates
        ['temperature', 'top-p', 'max-length'].forEach(param => {
            const slider = document.getElementById(param);
            const valueDisplay = document.getElementById(`${param}-value`);
            if (slider && valueDisplay) {
                slider.addEventListener('input', () => {
                    valueDisplay.textContent = slider.value;
                });
            }
        });
    }
}

// Translation functionality
class Translation {
    static displayTranslationPair(sourceText, targetText, sourceLang, targetLang, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = `
            <div class="translation-pair">
                <div class="source-text">
                    <div class="language-badge" style="background-color: var(--primary-color); color: white;">
                        ${sourceLang}
                    </div>
                    <p>${sourceText}</p>
                </div>
                <div class="target-text">
                    <div class="language-badge" style="background-color: var(--success-color); color: white;">
                        ${targetLang}
                    </div>
                    <p>${targetText}</p>
                </div>
            </div>
        `;
    }
    
    static createLanguageSelector(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const languages = [
            { code: 'en', name: 'English' },
            { code: 'es', name: 'Spanish' },
            { code: 'fr', name: 'French' },
            { code: 'de', name: 'German' },
            { code: 'ru', name: 'Russian' },
            { code: 'zh', name: 'Chinese' },
            { code: 'ar', name: 'Arabic' },
            { code: 'hi', name: 'Hindi' },
            { code: 'ja', name: 'Japanese' },
            { code: 'pt', name: 'Portuguese' },
            { code: 'it', name: 'Italian' }
        ];
        
        container.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <label for="source-lang" class="form-label">Source Language</label>
                    <select id="source-lang" class="form-select">
                        <option value="auto">Auto-detect</option>
                        ${languages.map(lang => `<option value="${lang.code}">${lang.name}</option>`).join('')}
                    </select>
                </div>
                <div class="col-md-6">
                    <label for="target-lang" class="form-label">Target Language</label>
                    <select id="target-lang" class="form-select">
                        ${languages.map(lang => `<option value="${lang.code}" ${lang.code === 'en' ? 'selected' : ''}>${lang.name}</option>`).join('')}
                    </select>
                </div>
            </div>
        `;
    }
}

// Classification functionality
class Classification {
    static displayResults(results, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = results.map(result => `
            <div class="classification-result">
                <div class="classification-label">${result.label}</div>
                <div class="classification-score" style="color: ${this.getScoreColor(result.score)};">
                    ${(result.score * 100).toFixed(1)}%
                </div>
            </div>
        `).join('');
    }
    
    static getScoreColor(score) {
        if (score > 0.7) return '#4CAF50';
        if (score > 0.4) return '#FF9800';
        return '#F44336';
    }
}

// Vector Embeddings functionality
class VectorEmbeddings {
    static displaySearchResults(results, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = results.map(result => `
            <div class="search-result">
                <div class="result-text">${result.text}</div>
                <div class="search-score">Similarity: ${(result.score * 100).toFixed(1)}%</div>
                <div class="progress mt-2" style="height: 8px;">
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${result.score * 100}%; background-color: ${this.getScoreColor(result.score)};">
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    static getScoreColor(score) {
        if (score > 0.7) return '#4CAF50';
        if (score > 0.4) return '#FF9800';
        return '#F44336';
    }
}

// Chart utilities
class ChartUtils {
    static createBarChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        
        const defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        };
        
        return new Chart(ctx, {
            type: 'bar',
            data: data,
            options: { ...defaultOptions, ...options }
        });
    }
    
    static createPieChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        
        const defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                }
            }
        };
        
        return new Chart(ctx, {
            type: 'pie',
            data: data,
            options: { ...defaultOptions, ...options }
        });
    }
    
    static createLineChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        
        const defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        };
        
        return new Chart(ctx, {
            type: 'line',
            data: data,
            options: { ...defaultOptions, ...options }
        });
    }
}

// Animation utilities
class AnimationUtils {
    static fadeIn(element, duration = 500) {
        element.style.opacity = '0';
        element.style.transition = `opacity ${duration}ms ease-in`;
        
        setTimeout(() => {
            element.style.opacity = '1';
        }, 10);
    }
    
    static slideIn(element, direction = 'left', duration = 500) {
        const transform = direction === 'left' ? 'translateX(-100%)' : 'translateX(100%)';
        element.style.transform = transform;
        element.style.transition = `transform ${duration}ms ease-out`;
        
        setTimeout(() => {
            element.style.transform = 'translateX(0)';
        }, 10);
    }
    
    static bounceIn(element, duration = 600) {
        element.style.transform = 'scale(0.3)';
        element.style.opacity = '0';
        element.style.transition = `all ${duration}ms ease-out`;
        
        setTimeout(() => {
            element.style.transform = 'scale(1)';
            element.style.opacity = '1';
        }, 10);
    }
}

// Export classes for global use
window.NLPComponents = {
    POSTagging,
    NER,
    SentimentAnalysis,
    TextGeneration,
    Translation,
    Classification,
    VectorEmbeddings,
    ChartUtils,
    AnimationUtils
};
