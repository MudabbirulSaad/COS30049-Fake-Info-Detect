const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function analyzeText(text) {
  const apiUrl = API_BASE_URL.endsWith('/api') ? `${API_BASE_URL}/predict` : `${API_BASE_URL}/api/predict`;
  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text,
      include_confidence: true
    })
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || 'Failed to analyze text');
  }

  const data = await response.json();
  
  if (!data.success) {
    throw new Error(data.message || 'Analysis failed');
  }

  return data.result;
}

export async function checkHealth() {
  const apiUrl = API_BASE_URL.endsWith('/api') ? `${API_BASE_URL}/health` : `${API_BASE_URL}/api/health`;
  const response = await fetch(apiUrl);
  
  if (!response.ok) {
    throw new Error('API health check failed');
  }

  return await response.json();
}

