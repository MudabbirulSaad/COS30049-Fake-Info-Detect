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
        <router-link to="/" class="nav-link">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
            <polyline points="9,22 9,12 15,12 15,22"/>
          </svg>
          Home
        </router-link>
        <a href="#" class="nav-link">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
          </svg>
          About
        </a>
        <a href="#" class="nav-link">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="3"/>
            <path d="M12 1v6m0 6v6"/>
            <path d="m21 12-6 0m-6 0-6 0"/>
          </svg>
          How it Works
        </a>
        <a href="#" class="nav-link">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
            <polyline points="22,6 12,13 2,6"/>
          </svg>
          Contact
        </a>
      </div>
    </div>

    <div class="mobile-overlay" :class="{ active: mobileMenuOpen }" @click="toggleMobileMenu"></div>

    <main class="main-content">
      <section class="result-section">
        <div class="container">
          <div class="result-card reliable">
            <div class="result-header">
              <div class="result-status-badge reliable">
                <div class="status-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20,6 9,17 4,12"/>
                  </svg>
                </div>
                <div class="status-text">
                  <span class="status-label">Verified</span>
                  <span class="status-description">Content appears reliable</span>
                </div>
              </div>
              <div class="result-summary">
                <h1 class="result-title">Analysis Complete</h1>
                <p class="result-subtitle">Our AI analysis suggests this content follows credible reporting patterns and contains verifiable information.</p>
              </div>
            </div>

            <div class="confidence-section">
              <div class="confidence-header">
                <h3>Confidence Score</h3>
                <span class="confidence-score" id="confidence-score">{{ confidencePercent }}%</span>
              </div>
              <div class="confidence-bar">
                <div class="confidence-track">
                  <div class="confidence-fill reliable" :style="{ width: confidencePercent + '%' }"></div>
                </div>
              </div>
              <p class="confidence-description">
                Based on language patterns, source credibility indicators, and factual consistency analysis.
              </p>
            </div>

            <div class="analysis-details">
              <h3>Key Findings</h3>
              <div class="findings-grid">
                <div class="finding-card positive">
                  <div class="finding-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M9 12l2 2 4-4"/>
                      <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"/>
                      <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"/>
                      <path d="M13 12h3"/>
                      <path d="M5 12h3"/>
                    </svg>
                  </div>
                  <div class="finding-content">
                    <h4>Credible Language</h4>
                    <p>Uses objective, factual language with proper attribution and neutral tone</p>
                  </div>
                </div>
                <div class="finding-card positive">
                  <div class="finding-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                      <polyline points="14,2 14,8 20,8"/>
                      <line x1="16" y1="13" x2="8" y2="13"/>
                      <line x1="16" y1="17" x2="8" y2="17"/>
                      <polyline points="10,9 9,9 8,9"/>
                    </svg>
                  </div>
                  <div class="finding-content">
                    <h4>Source Verification</h4>
                    <p>Contains verifiable claims with traceable references and citations</p>
                  </div>
                </div>
                <div class="finding-card positive">
                  <div class="finding-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                    </svg>
                  </div>
                  <div class="finding-content">
                    <h4>Balanced Reporting</h4>
                    <p>Maintains journalistic standards with balanced perspective and context</p>
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

            <div class="recommendations">
              <h3>Best Practices</h3>
              <div class="recommendation-cards">
                <div class="recommendation-card">
                  <div class="rec-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="12" cy="12" r="3"/>
                      <path d="M12 1v6m0 6v6"/>
                      <path d="m21 12-6 0m-6 0-6 0"/>
                    </svg>
                  </div>
                  <div class="rec-content">
                    <h4>Cross-Reference</h4>
                    <p>Always verify important information with multiple credible sources before sharing</p>
                  </div>
                </div>
                <div class="recommendation-card">
                  <div class="rec-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="12" cy="12" r="10"/>
                      <polyline points="12,6 12,12 16,14"/>
                    </svg>
                  </div>
                  <div class="rec-content">
                    <h4>Check Recency</h4>
                    <p>Verify the publication date and look for more recent updates on the topic</p>
                  </div>
                </div>
                <div class="recommendation-card">
                  <div class="rec-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                    </svg>
                  </div>
                  <div class="rec-content">
                    <h4>Consider Context</h4>
                    <p>Evaluate the source's perspective and potential bias when making decisions</p>
                  </div>
                </div>
              </div>
            </div>

            <div class="result-actions">
              <router-link to="/" class="btn primary">Analyze Another Text</router-link>
              <button class="btn secondary" @click="printReport">Save Report</button>
              <button class="btn secondary" @click="shareResult">Share Result</button>
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
              Advanced AI-powered misinformation detection to help you navigate 
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

function shareResult() {
  if (navigator.share) {
    navigator.share({
      title: 'Aura Analysis Result',
      text: 'Content verified as reliable by Aura AI'
    })
  }
}

onMounted(() => {
  const storedData = sessionStorage.getItem('analysisResult')

  if (!storedData) {
    router.push({ name: 'home' })
    return
  }

  try {
    const analysisData = JSON.parse(storedData)

    if (analysisData.prediction !== 'Reliable') {
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

