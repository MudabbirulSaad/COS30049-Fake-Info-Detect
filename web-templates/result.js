// Result page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    loadAnalysisResult();
    setupMobileMenu();
    setupInteractions();
});

// Load analysis result from sessionStorage
function loadAnalysisResult() {
    const analysisResult = sessionStorage.getItem('analysisResult');
    const analyzedText = sessionStorage.getItem('analyzedText');
    
    if (!analysisResult || !analyzedText) {
        // Redirect to home if no analysis data
        window.location.href = 'index.html';
        return;
    }
    
    try {
        const result = JSON.parse(analysisResult);
        populateResultData(result, analyzedText);
    } catch (error) {
        console.error('Error parsing analysis result:', error);
        window.location.href = 'index.html';
    }
}

// Populate the result page with analysis data
function populateResultData(result, text) {
    // Update confidence score
    const confidenceScore = document.getElementById('confidence-score');
    const confidenceFill = document.getElementById('confidence-fill');
    
    if (confidenceScore) {
        confidenceScore.textContent = `${result.confidence}%`;
    }
    
    // Animate confidence bar
    setTimeout(() => {
        if (confidenceFill) {
            confidenceFill.style.width = `${result.confidence}%`;
        }
    }, 500);
    
    // Update analyzed text
    const analyzedTextElement = document.getElementById('analyzed-text');
    if (analyzedTextElement) {
        // Truncate text if too long
        const maxLength = 500;
        const displayText = text.length > maxLength ? 
            text.substring(0, maxLength) + '...' : text;
        analyzedTextElement.textContent = displayText;
    }
    
    // Update analysis details based on result type
    updateAnalysisDetails(result);
}

// Update analysis details based on reliability
function updateAnalysisDetails(result) {
    const detailItems = document.querySelectorAll('.detail-item');
    
    if (result.isReliable) {
        // Update for reliable content
        updateDetailItem(detailItems[0], 'positive', '✓', 'Language Patterns', 
            'Uses objective, factual language with proper attribution');
        updateDetailItem(detailItems[1], 'positive', '✓', 'Source Indicators', 
            'Contains verifiable claims and references');
        updateDetailItem(detailItems[2], 'positive', '✓', 'Emotional Tone', 
            'Maintains neutral, informative tone');
    } else {
        // Update for unreliable content
        updateDetailItem(detailItems[0], 'negative', '⚠', 'Language Patterns', 
            'Contains emotionally charged or sensational language');
        updateDetailItem(detailItems[1], 'negative', '⚠', 'Source Indicators', 
            'Lacks proper attribution or verifiable sources');
        updateDetailItem(detailItems[2], 'negative', '⚠', 'Factual Consistency', 
            'May contain unsubstantiated claims or contradictions');
    }
}

// Update individual detail item
function updateDetailItem(item, type, icon, title, description) {
    if (!item) return;
    
    item.className = `detail-item ${type}`;
    
    const iconElement = item.querySelector('.detail-icon');
    const titleElement = item.querySelector('h4');
    const descElement = item.querySelector('p');
    
    if (iconElement) iconElement.textContent = icon;
    if (titleElement) titleElement.textContent = title;
    if (descElement) descElement.textContent = description;
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
    const mobileSidebar = document.getElementById('mobile-sidebar');
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

// Setup interactive elements
function setupInteractions() {
    // Add hover effects to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = '';
        });
    });
    
    // Smooth scrolling for internal links
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
}

// Share result functionality
function shareResult() {
    const result = sessionStorage.getItem('analysisResult');
    const text = sessionStorage.getItem('analyzedText');
    
    if (!result || !text) return;
    
    try {
        const analysisData = JSON.parse(result);
        const shareText = `I analyzed some text with Aura and found it to be ${analysisData.isReliable ? 'reliable' : 'unreliable'} with ${analysisData.confidence}% confidence.`;
        
        if (navigator.share) {
            navigator.share({
                title: 'Aura Analysis Result',
                text: shareText,
                url: window.location.href
            });
        } else {
            // Fallback: copy to clipboard
            navigator.clipboard.writeText(shareText + ' ' + window.location.href)
                .then(() => {
                    alert('Result copied to clipboard!');
                })
                .catch(() => {
                    alert('Unable to share. Please copy the URL manually.');
                });
        }
    } catch (error) {
        console.error('Error sharing result:', error);
        alert('Unable to share result.');
    }
}

// Report content functionality (for unreliable content)
function reportContent() {
    const text = sessionStorage.getItem('analyzedText');
    
    if (!text) return;
    
    // In a real application, this would send the content to a reporting system
    const confirmed = confirm('Report this content as potentially harmful misinformation?\n\nThis will help improve our detection algorithms.');
    
    if (confirmed) {
        // Simulate reporting
        setTimeout(() => {
            alert('Thank you for your report. We will review this content to improve our analysis.');
        }, 500);
        
        // In reality, you would send this to your backend:
        // fetch('/api/report', {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify({ text, timestamp: new Date().toISOString() })
        // });
    }
}

// Print functionality
function printReport() {
    window.print();
}

// Clear session data when leaving the page (optional)
window.addEventListener('beforeunload', function() {
    // Uncomment if you want to clear data when user navigates away
    // sessionStorage.removeItem('analysisResult');
    // sessionStorage.removeItem('analyzedText');
});

// Handle back button navigation
window.addEventListener('popstate', function() {
    // Clear session data if user goes back
    sessionStorage.removeItem('analysisResult');
    sessionStorage.removeItem('analyzedText');
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + P to print
    if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
        e.preventDefault();
        printReport();
    }
    
    // Escape to go back to home
    if (e.key === 'Escape') {
        window.location.href = 'index.html';
    }
});

// Add animation on page load
document.addEventListener('DOMContentLoaded', function() {
    const resultCard = document.querySelector('.result-card');
    if (resultCard) {
        resultCard.style.opacity = '0';
        resultCard.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            resultCard.style.transition = 'all 0.5s ease';
            resultCard.style.opacity = '1';
            resultCard.style.transform = 'translateY(0)';
        }, 100);
    }
});
