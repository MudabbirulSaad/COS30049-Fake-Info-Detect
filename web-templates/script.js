// DOM Elements
const textInput = document.getElementById('text-input');
const analyzeBtn = document.getElementById('analyze-btn');
const charCount = document.querySelector('.char-count');
const minChars = document.querySelector('.min-chars');
const loadingOverlay = document.getElementById('loading-overlay');

// State management
let isAnalyzing = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    updateCharCount();
    setupMobileMenu();
});

// Event Listeners
function setupEventListeners() {
    // Text input events
    textInput.addEventListener('input', handleTextInput);
    textInput.addEventListener('focus', handleInputFocus);
    textInput.addEventListener('blur', handleInputBlur);

    // Button events
    analyzeBtn.addEventListener('click', handleAnalyze);

    // Paste button
    const pasteBtn = document.getElementById('paste-btn');
    if (pasteBtn) {
        pasteBtn.addEventListener('click', handlePaste);
    }

    // Prevent form submission on Enter
    textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey && !analyzeBtn.disabled) {
            e.preventDefault();
            handleAnalyze();
        }
    });
}

// Setup mobile menu
function setupMobileMenu() {
    const mobileToggle = document.getElementById('mobile-menu-toggle');
    const mobileSidebar = document.getElementById('mobile-sidebar');
    const mobileOverlay = document.getElementById('mobile-overlay');

    if (mobileToggle && mobileSidebar && mobileOverlay) {
        // Toggle menu
        mobileToggle.addEventListener('click', function(e) {
            e.stopPropagation();
            toggleMobileMenu();
        });

        // Close menu when clicking overlay
        mobileOverlay.addEventListener('click', function() {
            closeMobileMenu();
        });

        // Close menu when clicking a nav link
        const mobileNavLinks = mobileSidebar.querySelectorAll('.nav-link');
        mobileNavLinks.forEach(link => {
            link.addEventListener('click', function() {
                closeMobileMenu();
            });
        });

        // Close menu on escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && mobileSidebar.classList.contains('active')) {
                closeMobileMenu();
            }
        });
    }
}

function toggleMobileMenu() {
    const mobileToggle = document.getElementById('mobile-menu-toggle');
    const mobileSidebar = document.getElementById('mobile-sidebar');
    const mobileOverlay = document.getElementById('mobile-overlay');

    const isActive = mobileSidebar.classList.contains('active');

    if (isActive) {
        closeMobileMenu();
    } else {
        openMobileMenu();
    }
}

function openMobileMenu() {
    const mobileToggle = document.getElementById('mobile-menu-toggle');
    const mobileSidebar = document.getElementById('mobile-sidebar');
    const mobileOverlay = document.getElementById('mobile-overlay');

    mobileToggle.classList.add('active');
    mobileSidebar.classList.add('active');
    mobileOverlay.classList.add('active');

    // Prevent body scroll
    document.body.style.overflow = 'hidden';
}

function closeMobileMenu() {
    const mobileToggle = document.getElementById('mobile-menu-toggle');
    const mobileSidebar = document.getElementById('mobile-sidebar');
    const mobileOverlay = document.getElementById('mobile-overlay');

    mobileToggle.classList.remove('active');
    mobileSidebar.classList.remove('active');
    mobileOverlay.classList.remove('active');

    // Restore body scroll
    document.body.style.overflow = '';
}

// Handle text input changes
function handleTextInput() {
    const text = textInput.value.trim();
    updateCharCount();
    updateButtonState(text);

    // Add active class when user is typing
    if (text.length > 0) {
        textInput.classList.add('active');
        minChars.classList.remove('show');
    } else {
        textInput.classList.remove('active');
        if (text.length < 10) {
            minChars.classList.add('show');
        }
    }
}

// Handle input focus
function handleInputFocus() {
    textInput.classList.add('active');
}

// Handle input blur
function handleInputBlur() {
    if (textInput.value.trim().length === 0) {
        textInput.classList.remove('active');
    }
}

// Update character count
function updateCharCount() {
    const count = textInput.value.length;
    charCount.textContent = `${count} character${count !== 1 ? 's' : ''}`;

    // Show/hide minimum character warning
    if (count > 0 && count < 10) {
        minChars.classList.add('show');
    } else {
        minChars.classList.remove('show');
    }
}

// Paste button functionality
async function handlePaste() {
    const pasteBtn = document.getElementById('paste-btn');

    try {
        // Check if clipboard API is available and we're in a secure context
        if (navigator.clipboard && window.isSecureContext) {
            try {
                const text = await navigator.clipboard.readText();
                if (text && text.trim()) {
                    textInput.value = text;
                    textInput.focus();
                    updateCharCount();
                    updateButtonState(text);

                    // Visual feedback
                    pasteBtn.innerHTML = `
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="20,6 9,17 4,12"/>
                        </svg>
                        Pasted
                    `;

                    setTimeout(() => {
                        pasteBtn.innerHTML = `
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                            </svg>
                            Paste
                        `;
                    }, 2000);
                    return;
                }
            } catch (clipboardError) {
                console.log('Clipboard access denied or failed:', clipboardError);
            }
        }

        // Fallback: Just focus the textarea for manual paste
        textInput.focus();
        textInput.select();

        // Show instruction to user
        pasteBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 12l2 2 4-4"/>
            </svg>
            Ctrl+V
        `;

        setTimeout(() => {
            pasteBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
                Paste
            `;
        }, 2000);

    } catch (err) {
        console.log('Paste operation failed:', err);
        // Final fallback: just focus the textarea
        textInput.focus();
        textInput.select();
    }
}

// Update button state based on input
function updateButtonState(text) {
    if (text.length >= 10 && !isAnalyzing) {
        analyzeBtn.disabled = false;
    } else {
        analyzeBtn.disabled = true;
    }
}

// Handle analyze button click
async function handleAnalyze() {
    if (isAnalyzing || analyzeBtn.disabled) return;

    const text = textInput.value.trim();
    if (text.length < 10) return;

    // Store text in sessionStorage for result page
    sessionStorage.setItem('analyzedText', text);

    // Start loading state
    setLoadingState(true);
    showLoadingOverlay();

    // Simulate analysis (replace with actual API call)
    try {
        const result = await simulateAnalysis(text);

        // Store result in sessionStorage
        sessionStorage.setItem('analysisResult', JSON.stringify(result));

        // Redirect to appropriate result page with text parameter
        const encodedText = encodeURIComponent(text);
        if (result.isReliable) {
            window.location.href = `reliable.html?text=${encodedText}`;
        } else {
            window.location.href = `unreliable.html?text=${encodedText}`;
        }
    } catch (error) {
        console.error('Analysis failed:', error);
        // Handle error state here
        setLoadingState(false);
        hideLoadingOverlay();
        alert('Analysis failed. Please try again.');
    }
}

// Simulate analysis with random results (replace with actual API)
function simulateAnalysis(text) {
    return new Promise((resolve) => {
        // Simulate processing time (2 seconds as requested)
        setTimeout(() => {
            // Simple heuristic for demo purposes
            const suspiciousWords = ['fake', 'hoax', 'conspiracy', 'secret', 'they dont want you to know', 'scam', 'lie', 'cover-up'];
            const lowerText = text.toLowerCase();
            const hasSuspiciousWords = suspiciousWords.some(word => lowerText.includes(word));

            // Random confidence between 70-95%
            const confidence = Math.floor(Math.random() * 26) + 70;

            const result = {
                isReliable: !hasSuspiciousWords && Math.random() > 0.3,
                confidence: confidence,
                reasoning: hasSuspiciousWords ?
                    'The text contains language patterns commonly associated with misinformation.' :
                    'The text appears to follow factual reporting patterns and contains verifiable claims.'
            };

            resolve(result);
        }, 2000); // Exactly 2 seconds as requested
    });
}

// Set loading state
function setLoadingState(loading) {
    isAnalyzing = loading;

    if (loading) {
        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;
        textInput.disabled = true;
    } else {
        analyzeBtn.classList.remove('loading');
        textInput.disabled = false;
        updateButtonState(textInput.value.trim());
    }
}

// Show loading overlay
function showLoadingOverlay() {
    loadingOverlay.classList.remove('hidden');

    // Animate progress bar
    const progressFill = loadingOverlay.querySelector('.progress-fill');
    const progressText = loadingOverlay.querySelector('.progress-text');

    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;

        progressFill.style.width = `${progress}%`;

        if (progress < 30) {
            progressText.textContent = 'Analyzing language patterns...';
        } else if (progress < 60) {
            progressText.textContent = 'Checking source credibility...';
        } else if (progress < 90) {
            progressText.textContent = 'Evaluating factual consistency...';
        } else {
            progressText.textContent = 'Finalizing analysis...';
        }
    }, 200);

    // Clear interval after 2 seconds
    setTimeout(() => {
        clearInterval(interval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Complete!';
    }, 1800);
}

// Hide loading overlay
function hideLoadingOverlay() {
    loadingOverlay.classList.add('hidden');
}

// Utility function to debounce input events
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

// Add some polish with micro-interactions
document.addEventListener('DOMContentLoaded', function() {
    // Add subtle hover effects to interactive elements
    const interactiveElements = document.querySelectorAll('button, textarea');

    interactiveElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            if (!this.disabled) {
                this.style.transform = 'translateY(-1px)';
            }
        });

        element.addEventListener('mouseleave', function() {
            this.style.transform = '';
        });
    });
});

// Handle keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !analyzeBtn.disabled) {
        e.preventDefault();
        handleAnalyze();
    }

    // Escape to hide loading overlay
    if (e.key === 'Escape' && !loadingOverlay.classList.contains('hidden')) {
        // Don't allow canceling analysis for now
        // Could add cancel functionality here
    }
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
