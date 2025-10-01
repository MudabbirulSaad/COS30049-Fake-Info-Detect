# Aura Misinformation Detection System

## Project Overview

The Aura Misinformation Detection System is a comprehensive machine learning solution for automated misinformation detection. This repository contains the complete pipeline including dataset preparation, model training and evaluation, hyperparameter tuning, a production-ready FastAPI backend, and a Vue.js web application. The system achieves 90.70% accuracy on the test set, exceeding the target threshold of 85%. The application is fully containerised using Docker for easy deployment across different platforms.

## Quick Start

**Docker Deployment (Recommended)**:
```bash
docker-compose build
docker-compose up -d
# Access at http://localhost
```

**Manual Deployment**:
```bash
# Backend
conda activate aura-misinformation
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev
# Access at http://localhost:5173
```

## Repository Structure

```
fake-info-detect/
├── dataset_preparation/               # Dataset preparation package
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration parameters
│   ├── data_loader.py                # Dataset loading utilities
│   ├── text_processor.py             # Text preprocessing module
│   ├── feature_engineer.py           # Feature engineering module
│   ├── quality_analyzer.py           # Quality assessment module
│   └── aura_dataset_preparation.py   # Main preparation pipeline
├── model_evaluation/                 # Model evaluation package
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration parameters
│   ├── data_handler.py               # Data loading and preprocessing
│   ├── model_trainer.py              # Model training procedures
│   ├── model_evaluator.py            # Model evaluation and metrics
│   ├── model_selector.py             # Model selection and saving
│   ├── hyperparameter_tuner.py       # Hyperparameter optimization
│   ├── tuning_pipeline.py            # Tuning pipeline
│   └── evaluation_pipeline.py        # Main evaluation pipeline
├── api/                              # FastAPI backend
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # API configuration
│   ├── main.py                       # FastAPI application
│   ├── schemas.py                    # Pydantic models
│   └── model_service.py              # Model loading and prediction
├── frontend/                         # Vue.js web application
│   ├── public/                      # Static assets
│   │   └── styles.css               # Application styles
│   ├── src/                         # Source code
│   │   ├── views/                   # Page components
│   │   │   ├── HomeView.vue         # Main analysis page
│   │   │   ├── ReliableView.vue     # Reliable result page
│   │   │   └── UnreliableView.vue   # Unreliable result page
│   │   ├── router/                  # Vue Router configuration
│   │   │   └── index.js             # Route definitions
│   │   ├── services/                # API integration
│   │   │   └── api.js               # API service functions
│   │   ├── App.vue                  # Root component
│   │   └── main.js                  # Application entry point
│   ├── package.json                 # Frontend dependencies
│   └── vite.config.js               # Vite configuration
├── web-templates/                    # Approved HTML templates
│   ├── index.html                   # Home page template
│   ├── reliable.html                # Reliable result template
│   └── unreliable.html              # Unreliable result template
├── datasets/                         # Raw dataset files
│   ├── LIAR/                        # LIAR dataset
│   │   ├── train.tsv
│   │   ├── test.tsv
│   │   └── valid.tsv
│   └── ISOT/                        # ISOT dataset
│       ├── True.csv
│       └── Fake.csv
├── output/                          # Dataset preparation outputs
│   ├── aura_processed_dataset.csv   # Final processed dataset
│   └── quality_analysis_report.json # Quality assessment report
├── models/                          # Model evaluation outputs
│   ├── best_model.joblib            # Best performing baseline model
│   ├── tuned_random_forest.joblib   # Hyperparameter-tuned model
│   ├── tfidf_vectorizer.joblib      # Fitted TF-IDF vectorizer
│   ├── tuned_tfidf_vectorizer.joblib # Vectorizer for tuned model
│   ├── model_evaluation_results.json # Baseline evaluation results
│   ├── hyperparameter_tuning_results.json # Tuning results
│   └── confusion_matrices/          # Visualization outputs
├── run_dataset_preparation.py       # Dataset preparation script
├── run_model_evaluation.py          # Model evaluation script
├── run_hyperparameter_tuning.py     # Hyperparameter optimization script
├── requirements-api.txt             # API dependencies
├── Dockerfile.backend               # Backend Docker configuration
├── Dockerfile.frontend              # Frontend Docker configuration
├── docker-compose.yml               # Multi-container orchestration
├── nginx.conf                       # Nginx configuration
├── .dockerignore                    # Docker build exclusions
├── Aura_Assignment_2_Report.md      # Academic project report
└── README.md                        # Documentation
```

## Installation and Setup

### Prerequisites

**Option 1: Docker Deployment (Recommended)**
- Docker Engine 20.10 or higher
- Docker Compose 2.0 or higher

**Option 2: Manual Installation**
- Python 3.9 or higher
- Conda package manager
- Node.js 16 or higher
- npm 8 or higher

### Docker Deployment (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/MudabbirulSaad/COS30049-Fake-Info-Detect
cd COS30049-Fake-Info-Detect
```

2. Ensure model files exist:
```bash
ls -lh models/
# Should show: tuned_random_forest.joblib, tuned_tfidf_vectorizer.joblib
```

3. Build and start services:
```bash
docker-compose build
docker-compose up -d
```

4. Verify deployment:
```bash
docker-compose ps
curl http://localhost:8000/api/health
```

5. Access application:
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

### Manual Installation

1. Create and activate conda environment:
```bash
conda create -n aura-misinformation python=3.9 -y
conda activate aura-misinformation
```

2. Install machine learning dependencies:
```bash
conda install pandas nltk scikit-learn matplotlib seaborn -y
pip install transformers torch textstat joblib
```

3. Install API dependencies (for backend):
```bash
pip install -r requirements-api.txt
```

Or install manually:
```bash
pip install fastapi uvicorn pydantic python-multipart
```

4. Install frontend dependencies:
```bash
cd frontend
npm install
```

### Dataset Preparation Execution

Execute the complete dataset preparation pipeline:

```bash
python run_dataset_preparation.py
```

The pipeline performs the following operations:
- Loads and standardizes LIAR and ISOT datasets
- Applies comprehensive text preprocessing procedures
- Generates TF-IDF vectorization and linguistic features
- Conducts quality assessment and validation
- Exports the processed dataset for model training

### Model Evaluation Execution

Execute the machine learning model evaluation pipeline:

```bash
python run_model_evaluation.py
```

The evaluation pipeline performs the following operations:
- Loads the processed dataset and creates train-test splits
- Trains multiple classification algorithms (Logistic Regression, Naive Bayes, Linear SVM, Random Forest)
- Evaluates each model using comprehensive metrics (Accuracy, Precision, Recall, F1-Score)
- Generates confusion matrices and detailed classification reports
- Selects the best performing model based on F1-Score
- Saves the best model and TF-IDF vectorizer for deployment

### Hyperparameter Tuning Execution

Execute the hyperparameter optimization pipeline for Random Forest:

```bash
python run_hyperparameter_tuning.py
```

The tuning pipeline performs the following operations:
- Loads the processed dataset and creates train-test splits
- Defines comprehensive hyperparameter grid for Random Forest optimization
- Executes GridSearchCV with 5-fold cross-validation to find optimal parameters
- Evaluates the tuned model on test data with comprehensive metrics
- Compares baseline vs tuned model performance
- Saves the optimized model and results for deployment

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Records | 50,648 |
| Reliable News | 27,720 (54.7%) |
| Unreliable News | 22,928 (45.3%) |
| Quality Score | 95/100 |
| File Size | ~100 MB |

## Model Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 90.58% | 90.62% | 90.58% | 90.56% |
| Linear SVM | 89.45% | 89.44% | 89.45% | 89.44% |
| Logistic Regression | 89.28% | 89.29% | 89.28% | 89.26% |
| Naive Bayes | 85.83% | 85.82% | 85.83% | 85.83% |

**Selected Model**: Random Forest (Best F1-Score: 90.56%)

## Hyperparameter Tuning Results

| Configuration | F1-Score | Improvement | Parameters |
|---------------|----------|-------------|------------|
| Baseline Random Forest | 90.56% | - | Default parameters |
| Tuned Random Forest | 90.68% | +0.13% | n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1 |

**Optimal Model**: Tuned Random Forest (F1-Score: 90.68%)

## Pipeline Components

### Data Processing Features
- Multi-source dataset integration (LIAR and ISOT)
- Binary label standardization for classification tasks
- Duplicate record identification and removal
- Comprehensive text preprocessing procedures
- Advanced feature engineering with TF-IDF and BERT embeddings

### Text Preprocessing Procedures
- Lowercase normalization
- URL and HTML tag removal
- Special character and punctuation handling
- Stopword removal using NLTK English corpus
- Lemmatization with WordNet

### Quality Assurance
- Automated quality assessment with scoring system
- Class distribution analysis and balance verification
- Vocabulary diversity evaluation
- Baseline model performance validation

## Research Objectives

- Target Classification Accuracy: 85% minimum threshold
- **Achieved Performance**: 90.68% F1-Score (Tuned Random Forest)
- Current Pipeline Status: Complete dataset preparation, model evaluation, and hyperparameter optimization
- Supported Model Types: Optimized Random Forest with TF-IDF features
- **Status**: Production-ready optimized model available for deployment

## Technical Implementation

### Core Technologies
- Python 3.9: Primary programming language
- pandas: Data manipulation and analysis
- NLTK: Natural language processing toolkit
- scikit-learn: Machine learning library
- transformers: BERT model implementation
- torch: Deep learning framework

### Architecture Design
- Modular package structure for maintainability
- Object-oriented design with specialized classes
- Comprehensive error handling and logging
- Reproducible results with fixed random seeds

## Usage Examples

### Loading Processed Dataset
```python
import pandas as pd

# Load the processed dataset
df = pd.read_csv('output/aura_processed_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

### Using Optimized Model for Predictions
```python
import joblib
import pandas as pd

# Load the tuned model and vectorizer
model = joblib.load('models/tuned_random_forest.joblib')
vectorizer = joblib.load('models/tuned_tfidf_vectorizer.joblib')

# Example prediction
text_sample = "Breaking news: Scientists discover new evidence..."
text_vectorized = vectorizer.transform([text_sample])
prediction = model.predict(text_vectorized)[0]
probability = model.predict_proba(text_vectorized)[0]

print(f"Prediction: {'Unreliable' if prediction == 1 else 'Reliable'}")
print(f"Confidence: {max(probability):.4f}")
```

### Model Comparison Testing
```python
# Compare baseline vs tuned model performance
python test_tuned_model.py
```

### Model Training Pipeline Example
```python
from model_evaluation import EvaluationPipeline

# Initialize and run complete evaluation
pipeline = EvaluationPipeline()
best_model_info = pipeline.run_complete_pipeline()

print(f"Best model: {best_model_info['model_name']}")
```

## Quality Validation

### Dataset Validation Results
- No missing values in final dataset
- Binary labels properly encoded (0 for reliable, 1 for unreliable)
- Text preprocessing successfully applied to all records
- Duplicate records identified and removed
- Dataset randomization confirmed

### Performance Metrics
- Quality Assessment Score: 95/100
- Baseline Model Accuracy: 85.8% (exceeds target threshold)
- Class Distribution: Well-balanced (54.7% vs 45.3%)
- Vocabulary Diversity: Appropriate for classification tasks

## FastAPI Backend

### Starting the API Server

1. Ensure the trained model files exist in the `models/` directory:
   - `tuned_random_forest.joblib`
   - `tuned_tfidf_vectorizer.joblib`

2. Start the FastAPI server:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

3. Access the API:
   - **API Base URL**: http://localhost:8000
   - **Interactive Docs**: http://localhost:8000/api/docs
   - **ReDoc**: http://localhost:8000/api/redoc

### API Endpoints

#### Health Check
```bash
GET /api/health
```
Check API and model availability.

#### Single Text Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "text": "Your text to analyze here",
  "include_confidence": true
}
```

#### Batch Prediction
```bash
POST /api/predict/batch
Content-Type: application/json

{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "include_confidence": true
}
```

#### Model Information
```bash
GET /api/model/info
```
Get model metadata and performance metrics.

### Testing the API

Run the comprehensive test suite:
```bash
python test_api.py
```

This will test all endpoints with sample data and display results.

### Using the API in Your Application

**Python Example:**
```python
import requests

# Analyze text
response = requests.post(
    "http://localhost:8000/api/predict",
    json={"text": "Your text here", "include_confidence": True}
)

result = response.json()
print(f"Prediction: {result['result']['prediction']}")
print(f"Confidence: {result['result']['confidence']:.2%}")
```

**JavaScript Example:**
```javascript
const response = await fetch('http://localhost:8000/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Your text here',
    include_confidence: true
  })
});

const data = await response.json();
console.log('Prediction:', data.result.prediction);
console.log('Confidence:', data.result.confidence);
```

## Web Application

### Starting the Complete Application

1. Start the FastAPI backend server:
```bash
conda activate aura-misinformation
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

2. Start the Vue.js frontend development server:
```bash
cd frontend
npm run dev
```

3. Access the application:
   - **Frontend**: http://localhost:5173
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/api/docs

### Web Application Features

The web application provides a user-friendly interface for misinformation detection:

- **Text Analysis Interface**: Input text for credibility assessment
- **Real-time Processing**: Immediate analysis with loading indicators
- **Detailed Results**: Comprehensive analysis with confidence scores
- **Result Pages**: Separate views for reliable and unreliable content
- **Responsive Design**: Mobile-friendly interface
- **Visual Feedback**: Color-coded results with detailed findings

### Application Architecture

- **Frontend**: Vue.js 3 with Vue Router for single-page application
- **Backend**: FastAPI with RESTful API endpoints
- **Communication**: Asynchronous HTTP requests for predictions
- **Styling**: Custom CSS based on approved design templates
- **State Management**: Vue Composition API with sessionStorage for secure data handling
- **Security**: Analysis results stored in sessionStorage, not exposed in URL parameters

### User Workflow

1. User enters or pastes text content on the home page
2. Application validates input (minimum 10 characters)
3. Text is sent to the backend API for analysis
4. Machine learning model processes the text
5. Analysis results are securely stored in sessionStorage
6. User is redirected to result page (reliable or unreliable)
7. Result page retrieves data from sessionStorage (not URL parameters)
8. Detailed analysis is displayed with confidence score
9. User can analyze additional text or save the report

### Security Features

- **No URL Parameter Exposure**: Analysis results are not exposed in URL parameters
- **SessionStorage Protection**: Results stored in browser sessionStorage, cleared on tab close
- **Route Guards**: Result pages redirect to home if no valid analysis data exists
- **Prediction Validation**: Result pages verify prediction type matches the page
- **Input Validation**: Backend validates all inputs before processing
- **CORS Protection**: API configured with specific allowed origins

## Model Performance Analysis

- **Optimized Model**: Tuned Random Forest Classifier
- **Final Accuracy**: 90.70% (exceeds 85% target)
- **Precision**: 90.74%
- **Recall**: 90.70%
- **F1-Score**: 90.68%
- **Hyperparameter Optimization**: +0.13% improvement over baseline
- **Model Configuration**: 300 estimators, unlimited depth, minimal leaf constraints
- **Inference Speed**: ~16ms average prediction time
- **Training Samples**: 50,648 articles from LIAR and ISOT datasets
- **Feature Space**: 10,000 TF-IDF features
- **Deployment Ready**: Production-optimized artifacts available in models/

## Project Structure

This dataset preparation pipeline provides:
- Modular and extensible codebase
- Comprehensive documentation and logging
- Reproducible preprocessing procedures
- Quality assessment and validation protocols