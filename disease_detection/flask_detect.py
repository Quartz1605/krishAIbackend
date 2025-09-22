import os
import sys
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging

# Add the disease_detection module path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

# Import RAG chatbot
from src.chatbot.disease_chatbot import DiseaseDetectionChatbot

# CNN model dependencies
import torch
import torchvision.transforms as transforms
from PIL import Image


class DiseaseDetector:
    """Disease Detection class that wraps the existing prediction functionality."""

    def __init__(self, model_path="model/plant_disease_cnn_scripted.pt"):
        """Initialize the disease detector with the trained model."""
        try:
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()

            self.train_classes = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]

            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])

        except Exception as e:
            raise Exception(f"Failed to load disease detection model: {str(e)}")

    def _preprocess_image(self, image_path):
        """Preprocess image for model prediction."""
        try:
            image = Image.open(image_path)
            img_tensor = self.transform(image).unsqueeze(0)
            return img_tensor
        except Exception as e:
            raise Exception(f"Failed to preprocess image: {str(e)}")

    def predict(self, image_path):
        """
        Predict disease from image path.

        Args:
            image_path (str): Path to the image file

        Returns:
            dict: Prediction results with disease, confidence, and sources
        """
        try:
            img_tensor = self._preprocess_image(image_path)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top_prob, top_indices = torch.topk(probabilities, 1)

                predicted_class = self.train_classes[top_indices[0].item()]
                confidence = top_prob[0].item()

            # Format disease name
            disease_parts = predicted_class.split('___')
            if len(disease_parts) > 1:
                plant_type = disease_parts[0].replace('_', ' ')
                disease_name = disease_parts[1].replace('_', ' ')
                if disease_name.lower() == 'healthy':
                    formatted_disease = f"Healthy {plant_type}"
                else:
                    formatted_disease = f"{plant_type} - {disease_name}"
            else:
                formatted_disease = predicted_class.replace('_', ' ')

            return {
                "disease": formatted_disease,
                "confidence": confidence,
                "raw_prediction": predicted_class,
                "sources": []
            }

        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Global detector instance
disease_detector = None


def initialize_disease_detector():
    """Initialize the disease detector instance with error handling."""
    global disease_detector
    try:
        model_paths = [
            str(Path(__file__).parent / "model" / "plant_disease_cnn_scripted.pt"),
            str(Path(__file__).parent.parent / "disease_detection" / "model" / "plant_disease_cnn_scripted.pt"),
            "model/plant_disease_cnn_scripted.pt"
        ]

        model_path = next((p for p in model_paths if Path(p).exists()), None)

        if model_path is None:
            raise FileNotFoundError("Could not find plant_disease_cnn_scripted.pt model file")

        logger.info(f"Initializing Disease Detector with model: {model_path}")
        disease_detector = DiseaseDetector(model_path=model_path)
        logger.info("Disease Detector initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize disease detector: {str(e)}")
        return False


@app.route('/imagehealth', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Disease Detection API is running",
        "detector_ready": disease_detector is not None
    }), 200


@app.route('/detect', methods=['POST'])
def detect_disease():
    try:
        if disease_detector is None:
            return jsonify({
                "success": False,
                "error": "Disease detector is not initialized."
            }), 500

        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided. Use key 'image'."
            }), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No image selected."
            }), 400

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        if not file.filename.lower().split('.')[-1] in allowed_extensions:
            return jsonify({
                "success": False,
                "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            }), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name

        logger.info(f"Processing uploaded image: {file.filename}")

        result = disease_detector.predict(temp_path)
        predicted_disease = result.get("disease", "Unknown")

        try:
            os.unlink(temp_path)
        except Exception:
            pass

        # Call RAG chatbot for advice
        advisor = DiseaseDetectionChatbot()
        advice = advisor.get_disease_solution(predicted_disease)

        return jsonify({
            "success": True,
            "disease": predicted_disease,
            "confidence": result.get("confidence", 0.0),
            "solution": advice.get("solution", "No advice available"),
            "sources": advice.get("sources", [])
        }), 200

    except Exception as e:
        logger.error(f"Error in detect_disease: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Failed to process image: {str(e)}"
        }), 500

# @app.route('/')
# def index():
#     return rern

@app.route('/imagedoc', methods=['GET'])
def home():
    return jsonify({
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "detect": "POST /detect",
            "home": "GET /"
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/detect"]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed",
        "tip": "Use POST for /detect, GET for /health and /"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": "Check server logs"
    }), 500


@app.errorhandler(413)
def payload_too_large(error):
    return jsonify({
        "success": False,
        "error": "File too large",
        "message": "Upload a smaller image"
    }), 413


def main():
    print("=" * 60)
    print("🌿 Plant Disease Detection API Server 🌿")
    print("=" * 60)

    if not initialize_disease_detector():
        print("\n❌ Failed to initialize disease detector.")
        sys.exit(1)

    print("\n🚀 Starting Flask server at http://127.0.0.1:4000")

    try:
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
        app.run(host='0.0.0.0', port=4000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")


if __name__ == '__main__':
    main()
