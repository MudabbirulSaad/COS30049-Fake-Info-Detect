"""
Pydantic models for request and response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class PredictRequest(BaseModel):
    """Request model for single text prediction."""
    
    text: str = Field(
        ...,
        description="Text content to analyze for misinformation",
        min_length=10,
        max_length=10000,
        example="Scientists at MIT have discovered a breakthrough in renewable energy technology."
    )
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence scores in the response"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text is not empty or only whitespace."""
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()


class PredictionResult(BaseModel):
    """Model for a single prediction result."""
    
    text: str = Field(..., description="Original text that was analyzed")
    prediction: str = Field(
        ...,
        description="Prediction result: 'Reliable' or 'Unreliable'"
    )
    confidence: float = Field(
        ...,
        description="Confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    processing_time_ms: float = Field(
        ...,
        description="Time taken to process the prediction in milliseconds"
    )


class PredictResponse(BaseModel):
    """Response model for single text prediction."""
    
    success: bool = Field(default=True, description="Whether the prediction was successful")
    result: PredictionResult = Field(..., description="Prediction result")


class BatchPredictRequest(BaseModel):
    """Request model for batch text prediction."""
    
    texts: List[str] = Field(
        ...,
        description="List of text contents to analyze",
        min_items=1,
        max_items=100,
        example=[
            "Breaking news: Scientists discover cure for all diseases!",
            "The Federal Reserve announced a 0.25% interest rate increase."
        ]
    )
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence scores in the response"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate all texts are not empty."""
        cleaned = []
        for text in v:
            if not text or not text.strip():
                raise ValueError('All texts must contain non-whitespace content')
            if len(text.strip()) < 10:
                raise ValueError('All texts must be at least 10 characters long')
            if len(text.strip()) > 10000:
                raise ValueError('All texts must be less than 10000 characters long')
            cleaned.append(text.strip())
        return cleaned


class BatchPredictResponse(BaseModel):
    """Response model for batch text prediction."""
    
    success: bool = Field(default=True, description="Whether the batch prediction was successful")
    total_processed: int = Field(..., description="Total number of texts processed")
    results: List[PredictionResult] = Field(..., description="List of prediction results")
    total_processing_time_ms: float = Field(
        ...,
        description="Total time taken to process all predictions in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="API status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    vectorizer_loaded: bool = Field(..., description="Whether the TF-IDF vectorizer is loaded")
    api_version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Response model for model information endpoint."""
    
    model_type: str = Field(..., description="Type of machine learning model")
    training_samples: int = Field(..., description="Number of samples used for training")
    accuracy: float = Field(..., description="Model accuracy on test set")
    precision: float = Field(..., description="Model precision on test set")
    recall: float = Field(..., description="Model recall on test set")
    f1_score: float = Field(..., description="Model F1-score on test set")
    features: int = Field(..., description="Number of TF-IDF features")
    hyperparameters: dict = Field(..., description="Model hyperparameters")


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    
    success: bool = Field(default=False, description="Always False for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Detailed error message")
    details: Optional[dict] = Field(None, description="Additional error details")

