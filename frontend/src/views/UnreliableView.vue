<template>
  <div>
    <header class="header">
      <nav class="navbar">
        <div class="nav-container">
          <router-link to="/" class="logo">
            <div class="logo-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"/>
                <path d="m21 21-4.35-4.35"/>
              </svg>
            </div>
            <span class="logo-text">Aura</span>
          </router-link>
          <div class="nav-links">
            <router-link to="/" class="nav-link">Home</router-link>
            <a href="#" class="nav-link">About</a>
            <a href="#" class="nav-link">How it Works</a>
            <a href="#" class="nav-link">Contact</a>
          </div>
          <button class="mobile-menu-toggle" @click="toggleMobileMenu">
            <span></span>
            <span></span>
            <span></span>
          </button>
        </div>
      </nav>
    </header>

    <div class="mobile-sidebar" :class="{ active: mobileMenuOpen }">
      <div class="nav-links">
        <router-link to="/" class="nav-link">Home</router-link>
        <a href="#" class="nav-link">About</a>
        <a href="#" class="nav-link">How it Works</a>
        <a href="#" class="nav-link">Contact</a>
      </div>
    </div>

    <div class="mobile-overlay" :class="{ active: mobileMenuOpen }" @click="toggleMobileMenu"></div>

    <main class="main-content">
      <section class="result-section">
        <div class="container">
          <div class="result-card unreliable">
            <div class="result-header">
              <div class="result-status-badge unreliable">
                <div class="status-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
                    <path d="M12 9v4"/>
                    <path d="M12 17h.01"/>
                  </svg>
                </div>
                <div class="status-text">
                  <span class="status-label">Flagged</span>
                  <span class="status-description">Content may be unreliable</span>
                </div>
              </div>
              <div class="result-summary">
                <h1 class="result-title">Analysis Complete</h1>
                <p class="result-subtitle">Our AI analysis detected patterns commonly associated with misinformation or unverified claims.</p>
              </div>
            </div>

            <div class="confidence-section">
              <div class="confidence-header">
                <h3>Confidence Score</h3>
                <span class="confidence-score" id="confidence-score">{{ confidencePercent }}%</span>
              </div>
              <div class="confidence-bar">
                <div class="confidence-track">
                  <div class="confidence-fill unreliable" :style="{ width: confidencePercent + '%' }"></div>
                </div>
              </div>
              <p class="confidence-description">
                Based on language patterns, emotional manipulation indicators, and factual inconsistency analysis.
              </p>
            </div>

            <div class="analysis-details">
              <h3>Red Flags Detected</h3>
              <div class="findings-grid">
                <div class="finding-card negative">
                  <div class="finding-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                      <line x1="12" y1="9" x2="12" y2="13"/>
                      <line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                  </div>
                  <div class="finding-content">
                    <h4>Suspicious Language</h4>
                    <p>Contains emotionally charged or sensational language patterns</p>
                  </div>
                </div>
                <div class="finding-card negative">
                  <div class="finding-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="12" cy="12" r="10"/>
                      <line x1="15" y1="9" x2="9" y2="15"/>
                      <line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                  </div>
                  <div class="finding-content">
                    <h4>Missing Sources</h4>
                    <p>Lacks proper attribution, citations, or verifiable references</p>
                  </div>
                </div>
                <div class="finding-card negative">
                  <div class="finding-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M9 12l2 2 4-4"/>
                      <circle cx="12" cy="12" r="10"/>
                      <path d="M9 12l2 2 4-4" stroke="none" fill="currentColor" opacity="0.3"/>
                    </svg>
                  </div>
                  <div class="finding-content">
                    <h4>Questionable Claims</h4>
                    <p>Contains unsubstantiated statements or potential contradictions</p>
                  </div>
                </div>
              </div>
            </div>

            <div class="analyzed-text-section">
              <h3>Analyzed Text</h3>
              <div class="analyzed-text-card">
                <p class="analyzed-text-content" id="analyzed-text">
                  {{ analyzedText }}
                </p>
              </div>
            </div>

            <div class="warning-notice">
              <div class="warning-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/>
                </svg>
              </div>
              <div class="warning-content">
                <h4>Exercise Caution</h4>
                <p>This content has been flagged as potentially unreliable. Please verify information through multiple credible sources before sharing or making decisions based on this content.</p>
              </div>
            </div>

            <div class="recommendations">
              <h3>Recommended Actions</h3>
              <div class="recommendation-cards">
                <div class="recommendation-card urgent">
                  <div class="rec-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="12" cy="12" r="10"/>
                      <line x1="15" y1="9" x2="9" y2="15"/>
                      <line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                  </div>
                  <div class="rec-content">
                    <h4>Do Not Share</h4>
                    <p>Avoid sharing this content without first verifying it through credible sources</p>
                  </div>
                </div>
                <div class="recommendation-card">
                  <div class="rec-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="11" cy="11" r="8"/>
                      <path d="m21 21-4.35-4.35"/>
                    </svg>
                  </div>
                  <div class="rec-content">
                    <h4>Fact-Check</h4>
                    <p>Cross-reference claims with established news organizations and fact-checking sites</p>
                  </div>
                </div>
                <div class="recommendation-card">
                  <div class="rec-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                      <polyline points="14,2 14,8 20,8"/>
                    </svg>
                  </div>
                  <div class="rec-content">
                    <h4>Find Sources</h4>
                    <p>Look for original sources, studies, or official statements that support the claims</p>
                  </div>
                </div>
                <div class="recommendation-card">
                  <div class="rec-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M9 12l2 2 4-4"/>
                      <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"/>
                      <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"/>
                      <path d="M13 12h3"/>
                      <path d="M5 12h3"/>
                    </svg>
                  </div>
                  <div class="rec-content">
                    <h4>Question Motives</h4>
                    <p>Consider who created this content and what they might gain from its spread</p>
                  </div>
                </div>
              </div>
            </div>

            <div class="result-actions">
              <router-link to="/" class="btn primary">Analyze Another Text</router-link>
              <button class="btn secondary" @click="printReport">Save Report</button>
              <button class="btn secondary" @click="reportContent">Report Content</button>
            </div>
          </div>
        </div>
      </section>
    </main>

    <footer class="footer">
      <div class="container">
        <div class="footer-content">
          <div class="footer-section">
            <div class="footer-logo">
              <div class="logo-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="11" cy="11" r="8"/>
                  <path d="m21 21-4.35-4.35"/>
                </svg>
              </div>
              <span class="logo-text">Aura</span>
            </div>
            <p class="footer-description">
              AI-powered misinformation detection to help you navigate 
              the digital information landscape with confidence.
            </p>
          </div>
          <div class="footer-section">
            <h4>Quick Links</h4>
            <ul class="footer-links">
              <li><a href="#">How it Works</a></li>
              <li><a href="#">Privacy Policy</a></li>
              <li><a href="#">Terms of Service</a></li>
              <li><a href="#">Contact Us</a></li>
            </ul>
          </div>
          <div class="footer-section">
            <h4>Resources</h4>
            <ul class="footer-links">
              <li><a href="#">Media Literacy</a></li>
              <li><a href="#">Fact-Checking Tips</a></li>
              <li><a href="#">API Documentation</a></li>
              <li><a href="#">Support</a></li>
            </ul>
          </div>
        </div>
        <div class="footer-bottom">
          <p>&copy; 2025 Aura. All rights reserved. • Always verify important information from multiple sources.</p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const mobileMenuOpen = ref(false)
const analyzedText = ref('')
const confidence = ref(0)
const processingTime = ref(0)

const confidencePercent = computed(() => {
  return Math.round(confidence.value * 100)
})

function toggleMobileMenu() {
  mobileMenuOpen.value = !mobileMenuOpen.value
}

function printReport() {
  window.print()
}

function reportContent() {
  alert('Content reporting functionality would be implemented here.')
}

onMounted(() => {
  const storedData = sessionStorage.getItem('analysisResult')

  if (!storedData) {
    router.push({ name: 'home' })
    return
  }

  try {
    const analysisData = JSON.parse(storedData)

    if (analysisData.prediction !== 'Unreliable') {
      router.push({ name: 'home' })
      return
    }

    analyzedText.value = analysisData.text
    confidence.value = analysisData.confidence
    processingTime.value = analysisData.processingTime
  } catch (error) {
    console.error('Failed to parse analysis data:', error)
    router.push({ name: 'home' })
  }
})
</script>

