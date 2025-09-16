// Main JavaScript for NLP Ultimate Tutorial

// Theme management
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update theme icon
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) {
        themeIcon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
}

// Initialize theme on page load
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) {
        themeIcon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
}

// Loading state management
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="text-center py-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Processing your request...</p>
            </div>
        `;
    }
}

function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element && element.innerHTML.includes('spinner-border')) {
        element.innerHTML = '';
    }
}

// Error handling
function showError(message, elementId = 'resultsContainer') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="alert alert-danger fade-in">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Error:</strong> ${message}
            </div>
        `;
    }
}

// Success message
function showSuccess(message, elementId = 'resultsContainer') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="alert alert-success fade-in">
                <i class="fas fa-check-circle"></i>
                <strong>Success:</strong> ${message}
            </div>
        `;
    }
}

// API request helper
async function makeApiRequest(url, data, method = 'POST') {
    try {
        const response = await fetch(url, {
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
        console.error('API request failed:', error);
        throw error;
    }
}

// Text processing functions
function processText(endpoint, text, additionalData = {}) {
    const data = { text: text, ...additionalData };
    
    showLoading('resultsContainer');
    
    makeApiRequest(endpoint, data)
        .then(response => {
            if (response.success) {
                displayResults(response.result);
            } else {
                showError(response.error || 'An error occurred while processing the text');
            }
        })
        .catch(error => {
            showError('Failed to process text: ' + error.message);
        })
        .finally(() => {
            hideLoading('resultsContainer');
        });
}

// Display results
function displayResults(result) {
    const container = document.getElementById('resultsContainer');
    if (container) {
        container.innerHTML = result;
        container.classList.add('fade-in');
    }
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        // Show temporary success message
        const toast = document.createElement('div');
        toast.className = 'alert alert-success position-fixed';
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 200px;';
        toast.innerHTML = '<i class="fas fa-check"></i> Copied to clipboard!';
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

// Download text as file
function downloadText(text, filename = 'nlp_result.txt') {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Format JSON for display
function formatJSON(obj) {
    return JSON.stringify(obj, null, 2);
}

// Create data table
function createDataTable(data, headers) {
    let table = '<div class="table-responsive"><table class="table table-striped table-hover">';
    
    // Header
    if (headers) {
        table += '<thead><tr>';
        headers.forEach(header => {
            table += `<th>${header}</th>`;
        });
        table += '</tr></thead>';
    }
    
    // Body
    table += '<tbody>';
    data.forEach(row => {
        table += '<tr>';
        if (Array.isArray(row)) {
            row.forEach(cell => {
                table += `<td>${cell}</td>`;
            });
        } else {
            Object.values(row).forEach(value => {
                table += `<td>${value}</td>`;
            });
        }
        table += '</tr>';
    });
    table += '</tbody></table></div>';
    
    return table;
}

// Create chart
function createChart(canvasId, type, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            }
        }
    };
    
    const chartOptions = { ...defaultOptions, ...options };
    
    return new Chart(ctx, {
        type: type,
        data: data,
        options: chartOptions
    });
}

// Smooth scroll to element
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Debounce function for input handling
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function for scroll handling
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Local storage helpers
function saveToStorage(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
        console.error('Failed to save to localStorage:', error);
    }
}

function loadFromStorage(key, defaultValue = null) {
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
        console.error('Failed to load from localStorage:', error);
        return defaultValue;
    }
}

// Session storage helpers
function saveToSession(key, value) {
    try {
        sessionStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
        console.error('Failed to save to sessionStorage:', error);
    }
}

function loadFromSession(key, defaultValue = null) {
    try {
        const item = sessionStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
        console.error('Failed to load from sessionStorage:', error);
        return defaultValue;
    }
}

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme
    initializeTheme();
    
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
    
    // Add click handlers for copy buttons
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('copy-btn')) {
            const text = e.target.getAttribute('data-copy');
            if (text) {
                copyToClipboard(text);
            }
        }
    });
    
    // Add click handlers for download buttons
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('download-btn')) {
            const text = e.target.getAttribute('data-download');
            const filename = e.target.getAttribute('data-filename') || 'nlp_result.txt';
            if (text) {
                downloadText(text, filename);
            }
        }
    });
    
    // Handle form submissions
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            // Handle form submission here
        });
    });
    
    // Add tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Export functions for global use
window.NLPUtils = {
    toggleTheme,
    showLoading,
    hideLoading,
    showError,
    showSuccess,
    makeApiRequest,
    processText,
    displayResults,
    copyToClipboard,
    downloadText,
    formatJSON,
    createDataTable,
    createChart,
    scrollToElement,
    debounce,
    throttle,
    saveToStorage,
    loadFromStorage,
    saveToSession,
    loadFromSession
};
