from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.inference import predict_tumor
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for detecting brain tumors in MRI scans using EfficientNet-B3",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://earlymed.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Handle CORS preflight requests
@app.middleware("http")
async def custom_cors_middleware(request, call_next):
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": request.headers.get("Origin", "*"),
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true"
        }
        return JSONResponse(content={}, status_code=200, headers=headers)
    
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    model_path = os.path.join("models", "Eff_net_b3_01_brain_tumor.pth")
    return {
        "status": "healthy",
        "version": app.version,
        "model_loaded": os.path.exists(model_path)
    }

# Predict tumor endpoint
@app.post("/predict-tumor")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        logger.error("Invalid file type uploaded")
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        # Read image file
        contents = await file.read()
        logger.info(f"Received image: {file.filename}, size: {len(contents)} bytes")
        
        # Perform prediction
        class_label, confidence = predict_tumor(contents)
        
        # Map class label to frontend-friendly format
        class_label = class_label.replace("_tumor", "").capitalize()
        if class_label == "No_tumor":
            class_label = "NoTumor"
        
        logger.info(f"Prediction: {class_label}, Confidence: {confidence:.4f}")
        return {
            "prediction": class_label,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Run the server (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8003)),
        log_level="info"
    )
