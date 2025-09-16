// API utilities for NLP Ultimate Tutorial Flask Application

class NLPAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.endpoints = {
            // Text processing endpoints
            preprocessing: '/api/preprocessing',
            tokenization: '/api/tokenization',
            posTagging: '/api/pos-tagging',
            namedEntity: '/api/named-entity',
            sentiment: '/api/sentiment',
            summarization: '/api/summarization',
            topicAnalysis: '/api/topic-analysis',
            questionAnswering: '/api/question-answering',
            textGeneration: '/api/text-generation',
            translation: '/api/translation',
            classification: '/api/classification',
            vectorEmbeddings: '/api/vector-embeddings',
            
            // Utility endpoints
            updateText: '/api/update_current_text',
            getText: '/api/get_current_text',
            textStatistics: '/api/text_statistics'
        };
    }
    
    // Generic API request method
    async request(endpoint, data = {}, method = 'POST') {
        try {
            const response = await fetch(this.baseUrl + endpoint, {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }
    
    // Text preprocessing
    async preprocessText(text, options = {}) {
        return await this.request(this.endpoints.preprocessing, {
            text: text,
            ...options
        });
    }
    
    // Tokenization
    async tokenizeText(text, tokenizerType = 'word') {
        return await this.request(this.endpoints.tokenization, {
            text: text,
            tokenizer_type: tokenizerType
        });
    }
    
    // POS Tagging
    async posTagText(text, taggerType = 'nltk') {
        return await this.request(this.endpoints.posTagging, {
            text: text,
            tagger_type: taggerType
        });
    }
    
    // Named Entity Recognition
    async recognizeEntities(text, modelType = 'spacy') {
        return await this.request(this.endpoints.namedEntity, {
            text: text,
            model_type: modelType
        });
    }
    
    // Sentiment Analysis
    async analyzeSentiment(text, analyzerType = 'vader') {
        return await this.request(this.endpoints.sentiment, {
            text: text,
            analyzer_type: analyzerType
        });
    }
    
    // Text Summarization
    async summarizeText(text, method = 'extractive', options = {}) {
        return await this.request(this.endpoints.summarization, {
            text: text,
            method: method,
            ...options
        });
    }
    
    // Topic Analysis
    async analyzeTopics(text, method = 'lda') {
        return await this.request(this.endpoints.topicAnalysis, {
            text: text,
            method: method
        });
    }
    
    // Question Answering
    async answerQuestion(context, question, options = {}) {
        return await this.request(this.endpoints.questionAnswering, {
            context: context,
            question: question,
            ...options
        });
    }
    
    // Text Generation
    async generateText(prompt, options = {}) {
        return await this.request(this.endpoints.textGeneration, {
            prompt: prompt,
            ...options
        });
    }
    
    // Translation
    async translateText(text, sourceLang = 'auto', targetLang = 'en') {
        return await this.request(this.endpoints.translation, {
            text: text,
            source_lang: sourceLang,
            target_lang: targetLang
        });
    }
    
    // Classification
    async classifyText(text, scenario = 'sentiment', options = {}) {
        return await this.request(this.endpoints.classification, {
            text: text,
            scenario: scenario,
            ...options
        });
    }
    
    // Vector Embeddings
    async getEmbeddings(text, query = '') {
        return await this.request(this.endpoints.vectorEmbeddings, {
            text: text,
            query: query
        });
    }
    
    // Utility methods
    async updateCurrentText(text) {
        return await this.request(this.endpoints.updateText, { text: text });
    }
    
    async getCurrentText() {
        return await this.request(this.endpoints.getText, {}, 'GET');
    }
    
    async getTextStatistics(text) {
        return await this.request(this.endpoints.textStatistics, { text: text });
    }
}

// Batch processing utility
class BatchProcessor {
    constructor(api) {
        this.api = api;
        this.queue = [];
        this.processing = false;
    }
    
    addTask(task) {
        this.queue.push(task);
        if (!this.processing) {
            this.processQueue();
        }
    }
    
    async processQueue() {
        this.processing = true;
        
        while (this.queue.length > 0) {
            const task = this.queue.shift();
            try {
                await task.execute();
                if (task.onSuccess) task.onSuccess(task.result);
            } catch (error) {
                if (task.onError) task.onError(error);
            }
        }
        
        this.processing = false;
    }
}

// Caching utility
class APICache {
    constructor(maxSize = 100) {
        this.cache = new Map();
        this.maxSize = maxSize;
    }
    
    get(key) {
        if (this.cache.has(key)) {
            const item = this.cache.get(key);
            // Move to end (most recently used)
            this.cache.delete(key);
            this.cache.set(key, item);
            return item;
        }
        return null;
    }
    
    set(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.maxSize) {
            // Remove least recently used item
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, value);
    }
    
    clear() {
        this.cache.clear();
    }
}

// Rate limiting utility
class RateLimiter {
    constructor(requestsPerMinute = 60) {
        this.requestsPerMinute = requestsPerMinute;
        this.requests = [];
    }
    
    async waitIfNeeded() {
        const now = Date.now();
        const oneMinuteAgo = now - 60000;
        
        // Remove old requests
        this.requests = this.requests.filter(time => time > oneMinuteAgo);
        
        if (this.requests.length >= this.requestsPerMinute) {
            const oldestRequest = Math.min(...this.requests);
            const waitTime = 60000 - (now - oldestRequest);
            if (waitTime > 0) {
                await new Promise(resolve => setTimeout(resolve, waitTime));
            }
        }
        
        this.requests.push(now);
    }
}

// Error handling utility
class ErrorHandler {
    static handle(error, context = '') {
        console.error(`Error in ${context}:`, error);
        
        let message = 'An unexpected error occurred';
        
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            message = 'Network error: Unable to connect to the server';
        } else if (error.message.includes('HTTP error')) {
            message = `Server error: ${error.message}`;
        } else if (error.message) {
            message = error.message;
        }
        
        return {
            success: false,
            error: message,
            context: context,
            timestamp: new Date().toISOString()
        };
    }
    
    static createErrorResponse(message, context = '') {
        return {
            success: false,
            error: message,
            context: context,
            timestamp: new Date().toISOString()
        };
    }
}

// Progress tracking utility
class ProgressTracker {
    constructor() {
        this.progress = 0;
        this.total = 0;
        this.callbacks = [];
    }
    
    setTotal(total) {
        this.total = total;
        this.progress = 0;
        this.notifyCallbacks();
    }
    
    increment(amount = 1) {
        this.progress += amount;
        this.notifyCallbacks();
    }
    
    setProgress(progress) {
        this.progress = progress;
        this.notifyCallbacks();
    }
    
    onProgress(callback) {
        this.callbacks.push(callback);
    }
    
    notifyCallbacks() {
        const percentage = this.total > 0 ? (this.progress / this.total) * 100 : 0;
        this.callbacks.forEach(callback => callback(percentage, this.progress, this.total));
    }
    
    reset() {
        this.progress = 0;
        this.total = 0;
        this.notifyCallbacks();
    }
}

// Export utilities
window.NLPAPI = NLPAPI;
window.BatchProcessor = BatchProcessor;
window.APICache = APICache;
window.RateLimiter = RateLimiter;
window.ErrorHandler = ErrorHandler;
window.ProgressTracker = ProgressTracker;
