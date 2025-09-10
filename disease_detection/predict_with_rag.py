import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.chatbot.disease_chatbot import DiseaseDetectionChatbot

# Load environment variables
load_dotenv()

# Load the trained model
model = torch.jit.load("model/plant_disease_cnn_scripted.pt", map_location="cpu")
model.eval()

# Class names from the trained model
train_classes = [
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


def preprocess_image(img_path):
    """
    Preprocess image for model prediction.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(img_path)
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor


def predict_image(img_tensor, model):
    """
    Predict disease from preprocessed image tensor.
    
    Args:
        img_tensor (torch.Tensor): Preprocessed image tensor
        model: Trained PyTorch model
        
    Returns:
        str: Predicted class name
    """
    yb = model(img_tensor)
    _, preds = torch.max(yb, dim=1)
    return train_classes[preds[0].item()]


def get_prediction_with_confidence(img_tensor, model):
    """
    Get prediction with confidence scores.
    
    Args:
        img_tensor (torch.Tensor): Preprocessed image tensor
        model: Trained PyTorch model
        
    Returns:
        tuple: (predicted_class, confidence_score, top_3_predictions)
    """
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        predictions = []
        for i in range(3):
            class_name = train_classes[top3_indices[i].item()]
            confidence = top3_prob[i].item()
            predictions.append((class_name, confidence))
        
        return predictions[0][0], predictions[0][1], predictions


class PlantDiseaseAnalyzer:
    """
    Complete plant disease analysis system with RAG-based solutions.
    """
    
    def __init__(self, db_path="./chroma_db", model_name=None):
        """
        Initialize the plant disease analyzer.
        
        Args:
            db_path (str): Path to ChromaDB database
            model_name (str): LLM model name ('gemini' or 'llama2')
        """
        self.db_path = db_path
        self.model_name = model_name
        self.chatbot = None
        
        # Initialize RAG chatbot
        try:
            print("🔧 Initializing RAG system...", file=sys.stderr)
            self.chatbot = DiseaseDetectionChatbot(
                db_path=db_path, 
                model_name=model_name
            )
            print("✅ RAG system ready!", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Warning: RAG system not available: {str(e)}", file=sys.stderr)
            self.chatbot = None
    
    def analyze_image(self, image_path, include_confidence=True, include_sources=True):
        """
        Complete analysis: image prediction + RAG-based solution.
        
        Args:
            image_path (str): Path to the image file
            include_confidence (bool): Include confidence scores
            include_sources (bool): Include source documents
            
        Returns:
            dict: Complete analysis results
        """
        try:
            # Step 1: Image preprocessing and prediction
            print("🔍 Analyzing image...", file=sys.stderr)
            img_tensor = preprocess_image(image_path)
            
            if include_confidence:
                prediction, confidence, top3 = get_prediction_with_confidence(img_tensor, model)
            else:
                prediction = predict_image(img_tensor, model)
                confidence = None
                top3 = None
            
            print(f"🔬 Disease prediction: {prediction}", file=sys.stderr)
            
            # Step 2: Generate solution using RAG
            solution_result = None
            if self.chatbot:
                print("💡 Generating treatment solution...", file=sys.stderr)
                solution_result = self.chatbot.get_disease_solution(
                    prediction, 
                    include_sources=include_sources
                )
            
            # Step 3: Compile results
            result = {
                "image_path": image_path,
                "prediction": {
                    "disease": prediction,
                    "confidence": confidence,
                    "top_3_predictions": top3
                },
                "solution": solution_result,
                "timestamp": None  # Can add timestamp if needed
            }
            
            return result
            
        except Exception as e:
            return {
                "image_path": image_path,
                "prediction": {
                    "disease": "Error",
                    "confidence": 0.0,
                    "top_3_predictions": None
                },
                "solution": {
                    "status": "error",
                    "solution": f"Analysis failed: {str(e)}"
                },
                "error": str(e)
            }
    
    def print_analysis_results(self, result):
        """
        Print analysis results in a formatted way.
        
        Args:
            result (dict): Analysis results from analyze_image()
        """
        print("\n" + "="*70)
        print("🌿 PLANT DISEASE ANALYSIS RESULTS 🌿")
        print("="*70)
        
        # Image info
        print(f"📷 Image: {Path(result['image_path']).name}")
        
        # Prediction results
        prediction = result['prediction']
        print(f"\n🔬 Disease Detection:")
        print(f"   Primary Diagnosis: {prediction['disease']}")
        
        if prediction['confidence']:
            print(f"   Confidence: {prediction['confidence']:.1%}")
            
            if prediction['top_3_predictions']:
                print(f"\n   Top 3 Predictions:")
                for i, (disease, conf) in enumerate(prediction['top_3_predictions'], 1):
                    print(f"      {i}. {disease} ({conf:.1%})")
        
        # Solution results
        solution = result.get('solution')
        if solution and solution.get('status') != 'error':
            print(f"\n🏥 Analysis:")
            print(f"   Plant Type: {solution.get('plant_type', 'Unknown')}")
            print(f"   Disease: {solution.get('disease_name', 'Unknown')}")
            print(f"   Status: {solution.get('status', 'Unknown').upper()}")
            print(f"   Advice Type: {solution.get('solution_type', 'Unknown').title()}")
            
            print(f"\n💡 {solution.get('solution_type', 'Treatment').title()} Recommendations:")
            print(f"{solution.get('solution', 'No solution available.')}")
            
            # Sources
            sources = solution.get('sources', [])
            if sources:
                print(f"\n📚 Knowledge Sources ({len(sources)}):")
                for i, source in enumerate(sources, 1):
                    print(f"   {i}. {source['filename']}")
        else:
            print(f"\n❌ Solution Generation: Failed")
            if solution:
                print(f"   Error: {solution.get('solution', 'Unknown error')}")
        
        print("="*70)
    
    def interactive_analysis(self, image_path):
        """
        Run interactive analysis with follow-up Q&A.
        
        Args:
            image_path (str): Path to the image file
        """
        # Get initial analysis
        result = self.analyze_image(image_path)
        self.print_analysis_results(result)
        
        # Interactive follow-up if RAG is available
        if self.chatbot and result.get('solution') and result['solution'].get('status') != 'error':
            prediction = result['prediction']['disease']
            self.chatbot.chat_with_prediction(prediction)


def analyze_single_image(image_path, db_path="./chroma_db", model_name=None):
    """
    Quick function to analyze a single image.
    
    Args:
        image_path (str): Path to the image file
        db_path (str): Path to ChromaDB database
        model_name (str): LLM model name
        
    Returns:
        dict: Analysis results
    """
    analyzer = PlantDiseaseAnalyzer(db_path=db_path, model_name=model_name)
    return analyzer.analyze_image(image_path)


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection with RAG-based Solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_with_rag.py --image virus4.JPG
  python predict_with_rag.py --image virus4.JPG --interactive
  python predict_with_rag.py --image virus4.JPG --model gemini
  python predict_with_rag.py --image virus4.JPG --no-rag
        """
    )
    
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="Path to ChromaDB database")
    parser.add_argument("--model", type=str, default=None, help="LLM model: 'gemini' or 'llama2'")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode with follow-up Q&A")
    parser.add_argument("--no-rag", action="store_true", help="Skip RAG solution generation")
    parser.add_argument("--no-sources", action="store_true", help="Don't include source documents")
    
    args = parser.parse_args()
    
    try:
        # Check if image exists
        if not Path(args.image).exists():
            print(f"❌ Error: Image file '{args.image}' not found.")
            sys.exit(1)
        
        # Create analyzer
        analyzer = None if args.no_rag else PlantDiseaseAnalyzer(
            db_path=args.db_path, 
            model_name=args.model
        )
        
        if args.no_rag:
            # Simple prediction only
            print("🔍 Analyzing image (prediction only)...")
            img_tensor = preprocess_image(args.image)
            prediction, confidence, top3 = get_prediction_with_confidence(img_tensor, model)
            
            print(f"\n🔬 Disease Prediction: {prediction}")
            print(f"📊 Confidence: {confidence:.1%}")
            print(f"\n📋 Top 3 Predictions:")
            for i, (disease, conf) in enumerate(top3, 1):
                print(f"   {i}. {disease} ({conf:.1%})")
        else:
            # Full analysis with RAG
            if args.interactive:
                analyzer.interactive_analysis(args.image)
            else:
                result = analyzer.analyze_image(args.image, include_sources=not args.no_sources)
                analyzer.print_analysis_results(result)
    
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted. Goodbye! 🌱")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
