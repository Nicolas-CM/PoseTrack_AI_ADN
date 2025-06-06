"""
Real-time Activity Classification Module
"""

import numpy as np
import joblib
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json

from config.settings import MODEL_CONFIG, ACTIVITIES, MODELS_PATH


class ActivityClassifier:
    """Human activity classifier for real-time predictions"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_name = None
        self.feature_scaler = None
        self.label_encoder = None
        self.model_info = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from file
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if the model is loaded successfully, False otherwise
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                print(f"Error: File {model_path} does not exist")
                return False
            
            # Load the model
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_scaler = model_data.get('scaler')
                self.label_encoder = model_data.get('label_encoder')
                self.model_info = model_data.get('info', {})
                self.model_name = self.model_info.get('model_type', 'unknown')
            else:
                # Backward compatibility for legacy models
                self.model = model_data
                self.model_name = 'legacy'
            
            print(f"Model {self.model_name} loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict activity from a feature vector
        
        Args:
            features: Feature array
            
        Returns:
            Tuple with (predicted_activity, confidence, class_probabilities)
        """
        if self.model is None:
            return "no_model", 0.0, {}
        
        try:
            # Ensure features are 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Apply normalization if available
            if self.feature_scaler is not None:
                features = self.feature_scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            probabilities = {}
            confidence = 0.0
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = np.max(proba)
                
                if self.label_encoder is not None:
                    classes = self.label_encoder.classes_
                else:
                    classes = list(ACTIVITIES.keys())
                
                for i, class_name in enumerate(classes):
                    if i < len(proba):
                        probabilities[class_name] = float(proba[i])
            else:
                confidence = 0.8
                probabilities[prediction] = confidence
            
            if self.label_encoder is not None:
                try:
                    predicted_activity = self.label_encoder.inverse_transform([prediction])[0]
                except:
                    predicted_activity = str(prediction)
            else:
                predicted_activity = str(prediction)
            
            return predicted_activity, confidence, probabilities
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0, {}
    
    def predict_batch(self, features_batch: np.ndarray) -> List[Tuple[str, float]]:
        """
        Predict activities for a batch of feature vectors
        
        Args:
            features_batch: 2D array with features of multiple samples
            
        Returns:
            List of tuples (activity, confidence)
        """
        if self.model is None:
            return [("no_model", 0.0)] * len(features_batch)
        
        results = []
        
        try:
            if self.feature_scaler is not None:
                features_batch = self.feature_scaler.transform(features_batch)
            
            predictions = self.model.predict(features_batch)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_batch)
                confidences = np.max(probabilities, axis=1)
            else:
                confidences = [0.8] * len(predictions)
            
            for pred, conf in zip(predictions, confidences):
                if self.label_encoder is not None:
                    try:
                        activity = self.label_encoder.inverse_transform([pred])[0]
                    except:
                        activity = str(pred)
                else:
                    activity = str(pred)
                
                results.append((activity, float(conf)))
            
        except Exception as e:
            print(f"Batch prediction error: {e}")
            results = [("error", 0.0)] * len(features_batch)
        
        return results
    
    def get_activity_description(self, activity_code: str) -> str:
        """
        Get a human-readable description of an activity
        
        Args:
            activity_code: Activity code
            
        Returns:
            Activity description
        """
        return ACTIVITIES.get(activity_code, f"Activity: {activity_code}")
    
    def get_model_info(self) -> Dict:
        """
        Return information about the currently loaded model
        
        Returns:
            Dictionary with model info
        """
        info = {
            'model_loaded': self.model is not None,
            'model_name': self.model_name,
            'has_scaler': self.feature_scaler is not None,
            'has_label_encoder': self.label_encoder is not None,
            'supported_activities': list(ACTIVITIES.keys())
        }
        
        info.update(self.model_info)
        return info
    
    def is_ready(self) -> bool:
        """
        Check if classifier is ready for predictions
        
        Returns:
            True if model is loaded
        """
        return self.model is not None
    
    def get_available_models(self) -> List[Dict]:
        """
        Get list of available models in the models directory
        
        Returns:
            List of dicts with info of available models
        """
        models = []
        models_dir = Path(MODELS_PATH)
        
        if not models_dir.exists():
            return models
        
        for model_file in models_dir.glob("*.pkl"):
            try:
                model_data = joblib.load(model_file)
                
                if isinstance(model_data, dict) and 'info' in model_data:
                    info = model_data['info']
                    models.append({
                        'path': str(model_file),
                        'name': model_file.stem,
                        'type': info.get('model_type', 'unknown'),
                        'accuracy': info.get('accuracy', 'N/A'),
                        'created': info.get('created_at', 'N/A'),
                        'features': info.get('n_features', 'N/A')
                    })
                else:
                    models.append({
                        'path': str(model_file),
                        'name': model_file.stem,
                        'type': 'legacy',
                        'accuracy': 'N/A',
                        'created': 'N/A',
                        'features': 'N/A'
                    })
                    
            except Exception as e:
                print(f"Error reading model {model_file}: {e}")
        
        return models
    
    def clear_model(self):
        """Clear the currently loaded model"""
        self.model = None
        self.model_name = None
        self.feature_scaler = None
        self.label_encoder = None
        self.model_info = {}


class ActivityBuffer:
    """Buffer to smooth activity predictions
    
    This class implements a temporal smoothing buffer that maintains a history of recent
    predictions and their confidence levels. It helps stabilize activity recognition
    by filtering out momentary misclassifications and selecting the most consistent
    prediction over a time window.
    """
    
    def __init__(self, buffer_size: int = 10, confidence_threshold: float = 0.6):
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.predictions = []
        self.confidences = []
    
    def add_prediction(self, activity: str, confidence: float):
        """
        Add a new prediction to the buffer
        
        Args:
            activity: Predicted activity
            confidence: Prediction confidence
        """
        self.predictions.append(activity)
        self.confidences.append(confidence)
        
        if len(self.predictions) > self.buffer_size:
            self.predictions.pop(0)
            self.confidences.pop(0)
    def get_smoothed_prediction(self) -> Tuple[str, float]:
        """
        Get the smoothed prediction based on recent history
        
        This method analyzes the buffer of recent predictions to determine
        the most consistent activity. It counts occurrences of each activity
        that meets the confidence threshold and returns the most frequent one
        with its average confidence score.
        
        Returns:
            Tuple with (smoothed_activity, average_confidence)
        """
        if not self.predictions:
            return "unknown", 0.0
        
        activity_counts = {}
        activity_confidences = {}
        
        for activity, confidence in zip(self.predictions, self.confidences):
            if confidence >= self.confidence_threshold:
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
                if activity not in activity_confidences:
                    activity_confidences[activity] = []
                activity_confidences[activity].append(confidence)
        
        if not activity_counts:
            return "uncertain", np.mean(self.confidences)
        
        most_common = max(activity_counts.items(), key=lambda x: x[1])
        best_activity = most_common[0]
        avg_confidence = np.mean(activity_confidences[best_activity])
        
        return best_activity, avg_confidence
    
    def clear(self):
        """Clear the buffer"""
        self.predictions.clear()
        self.confidences.clear()
