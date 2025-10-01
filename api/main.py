"""
Main FastAPI application for Aura Misinformation Detection API.
"""

import logging
import time
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    CORS_ORIGINS,
    MODEL_INFO
)
from .schemas import (
    PredictRequest,
    PredictResponse,
    PredictionResult,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from .model_service import model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load the model and vectorizer on application startup."""
    logger.info("Starting Aura Misinformation Detection API")
    logger.info("Loading machine learning model...")
    
    success = model_service.load_model()
    
    if success:
        logger.info("Model loaded successfully. API is ready.")
    else:
        logger.error("Failed to load model. API will return errors for predictions.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Aura Misinformation Detection API")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Aura Misinformation Detection API",
        "version": API_VERSION,
        "docs": "/api/docs",
        "health": "/api/health"
    }


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check the health status of the API and model availability"
)
async def health_check():
    """
    Health check endpoint to verify API and model status.
    
    Returns:
        HealthResponse: API health status and model availability
    """
    return HealthResponse(
        status="healthy" if model_service.is_ready() else "unhealthy",
        model_loaded=model_service.model_loaded,
        vectorizer_loaded=model_service.vectorizer_loaded,
        api_version=API_VERSION
    )


@app.get(
    "/api/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Model Information",
    description="Get detailed information about the trained model"
)
async def get_model_info():
    """
    Get information about the trained misinformation detection model.
    
    Returns:
        ModelInfoResponse: Model metadata and performance metrics
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please check API health."
        )
    
    return ModelInfoResponse(**MODEL_INFO)


@app.post(
    "/api/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Analyze Single Text",
    description="Analyze a single text sample for misinformation detection",
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Model not available"}
    }
)
async def predict(request: PredictRequest):
    """
    Analyze a single text sample for misinformation.
    
    Args:
        request: PredictRequest containing text to analyze
        
    Returns:
        PredictResponse: Prediction result with confidence score
        
    Raises:
        HTTPException: If model is not ready or prediction fails
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please check API health."
        )
    
    try:
        prediction, confidence, processing_time = model_service.predict_single(
            request.text,
            request.include_confidence
        )
        
        result = PredictionResult(
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            prediction=prediction,
            confidence=confidence,
            processing_time_ms=processing_time
        )
        
        return PredictResponse(success=True, result=result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/api/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    summary="Analyze Multiple Texts",
    description="Analyze multiple text samples simultaneously for misinformation detection",
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Model not available"}
    }
)
async def predict_batch(request: BatchPredictRequest):
    """
    Analyze multiple text samples for misinformation.
    
    Args:
        request: BatchPredictRequest containing list of texts to analyze
        
    Returns:
        BatchPredictResponse: List of prediction results
        
    Raises:
        HTTPException: If model is not ready or prediction fails
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please check API health."
        )
    
    try:
        start_time = time.time()
        
        predictions = model_service.predict_batch(
            request.texts,
            request.include_confidence
        )
        
        results = []
        for text, (prediction, confidence, proc_time) in zip(request.texts, predictions):
            result = PredictionResult(
                text=text[:100] + "..." if len(text) > 100 else text,
                prediction=prediction,
                confidence=confidence,
                processing_time_ms=proc_time
            )
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictResponse(
            success=True,
            total_processed=len(results),
            results=results,
            total_processing_time_ms=total_time
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.__class__.__name__,
            "message": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Custom exception handler for general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "InternalServerError",
            "message": "An unexpected error occurred"
        }
    )

