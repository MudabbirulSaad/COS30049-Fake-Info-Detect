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
            <a href="#" class="nav-link active">Home</a>
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
        <a href="#" class="nav-link active">Home</a>
        <a href="#" class="nav-link">About</a>
        <a href="#" class="nav-link">How it Works</a>
        <a href="#" class="nav-link">Contact</a>
      </div>
    </div>

    <div class="mobile-overlay" :class="{ active: mobileMenuOpen }" @click="toggleMobileMenu"></div>

    <main class="main-content">
      <section class="hero">
        <div class="container">
          <div class="hero-content">
            <h1 class="hero-title">Detect Misinformation with Confidence</h1>
            <p class="hero-subtitle">
              Use AI analysis to verify the credibility of news articles,
              social media posts, and any text content in seconds.
            </p>
          </div>
        </div>
      </section>

      <section class="analysis-section">
        <div class="container">
          <div class="analysis-card">
            <div class="input-section">
              <label for="text-input" class="input-label">
                Paste or type the text you want to analyze
              </label>
              <div class="input-container">
                <textarea
                  id="text-input"
                  v-model="inputText"
                  class="text-input"
                  placeholder="Enter the text, article, or claim you'd like to fact-check..."
                  rows="8"
                  @input="updateCharCount"
                ></textarea>
                <button class="paste-btn" @click="pasteFromClipboard" title="Click to paste from clipboard">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                  </svg>
                  Paste
                </button>
              </div>
              <div class="input-footer">
                <span class="char-count">{{ charCount }} characters</span>
                <span class="min-chars">Minimum 10 characters required</span>
              </div>
            </div>

            <button 
              id="analyze-btn" 
              class="analyze-btn" 
              :class="{ loading: isAnalyzing }"
              :disabled="!canAnalyze || isAnalyzing"
              @click="analyzeText"
            >
              <span class="btn-text">Analyze Text</span>
              <div class="btn-loader">
                <div class="spinner"></div>
              </div>
            </button>

            <div v-if="errorMessage" style="margin-top: 1rem; padding: 1rem; background-color: #FEE2E2; border: 1px solid #EF4444; border-radius: 8px; color: #991B1B;">
              {{ errorMessage }}
            </div>
          </div>
        </div>
      </section>

      <div id="loading-overlay" class="loading-overlay" :class="{ hidden: !isAnalyzing }">
        <div class="loading-content">
          <div class="loading-spinner">
            <div class="spinner-large"></div>
          </div>
          <h3 class="loading-title">Analyzing Content...</h3>
          <p class="loading-text">Our AI is examining the text for credibility indicators</p>
          <div class="loading-progress">
            <div class="progress-bar">
              <div class="progress-fill"></div>
            </div>
            <span class="progress-text">Processing...</span>
          </div>
        </div>
      </div>
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
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { analyzeText as apiAnalyzeText } from '../services/api'

const router = useRouter()
const inputText = ref('')
const charCount = ref(0)
const isAnalyzing = ref(false)
const errorMessage = ref('')
const mobileMenuOpen = ref(false)

const canAnalyze = computed(() => {
  return inputText.value.trim().length >= 10
})

function updateCharCount() {
  charCount.value = inputText.value.length
}

function toggleMobileMenu() {
  mobileMenuOpen.value = !mobileMenuOpen.value
}

async function pasteFromClipboard() {
  try {
    const text = await navigator.clipboard.readText()
    inputText.value = text
    updateCharCount()
  } catch (err) {
    console.error('Failed to read clipboard:', err)
  }
}

async function analyzeText() {
  if (!canAnalyze.value || isAnalyzing.value) {
    return
  }

  errorMessage.value = ''
  isAnalyzing.value = true

  try {
    const result = await apiAnalyzeText(inputText.value)

    const routeName = result.prediction === 'Reliable' ? 'reliable' : 'unreliable'

    const analysisData = {
      text: inputText.value,
      prediction: result.prediction,
      confidence: result.confidence,
      processingTime: result.processing_time_ms,
      timestamp: new Date().toISOString()
    }

    sessionStorage.setItem('analysisResult', JSON.stringify(analysisData))

    router.push({
      name: routeName
    })
  } catch (error) {
    errorMessage.value = error.message || 'Failed to analyze text. Please try again.'
    isAnalyzing.value = false
  }
}
</script>

