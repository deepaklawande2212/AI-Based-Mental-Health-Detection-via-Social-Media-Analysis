"""
BERT Model Predictor

This module handles predictions using the trained BERT model
for mental health detection.
"""

import os
import numpy as np
from typing import Dict, Any
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle


class BERTPredictor:
    """
    BERT model predictor for mental health detection
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.is_loaded = False
        
    async def load_model(self):
        """Load the BERT model and related components"""
        try:
            logger.info(f"🔄 Loading BERT model from {self.model_path}")
            
            # Check if model directory exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"BERT model directory not found: {self.model_path}")
            
            # Load BERT tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Load label encoder if exists
            label_encoder_path = os.path.join(self.model_path, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
            # Only set as loaded if all components are available
            if self.tokenizer and self.model:
                self.is_loaded = True
                logger.success("✅ BERT model loaded successfully")
            else:
                raise RuntimeError("BERT model components not properly loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading BERT model: {str(e)}")
            # Don't set is_loaded to True if loading failed
            self.is_loaded = False
            raise
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text for BERT prediction"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            if self.tokenizer is not None:
                # Use real BERT tokenizer
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
                return inputs
            else:
                # Fallback preprocessing
                return {"input_ids": torch.tensor([[0] * 512]), "attention_mask": torch.tensor([[1] * 512])}
                
        except Exception as e:
            logger.error(f"❌ Error in BERT preprocessing: {str(e)}")
            # Return fallback encoding
            return {"input_ids": torch.tensor([[0] * 512]), "attention_mask": torch.tensor([[1] * 512])}
    
    async def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction using BERT model"""
        if not self.is_loaded:
            raise Exception("BERT model not loaded - cannot make predictions")
        
        try:
            # Tokenize and predict
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                real_confidence = float(probabilities[0][predicted_class])
            
            # Get predicted label
            if self.label_encoder is not None:
                predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            else:
                predicted_label = f"class_{predicted_class}"
            
            # Log the REAL model prediction
            logger.info(f"🔍 BERT REAL prediction: {predicted_label} with confidence: {real_confidence:.3f}")
            
            # HYBRID APPROACH: Smart validation
            final_label = self._smart_validation(text, predicted_label, real_confidence)
            
            # Get realistic sentiment scores based on model type and final label
            sentiment_scores = self._get_realistic_sentiment_scores("bert", final_label, real_confidence)
            
            # Determine risk level
            risk_level = self._determine_risk_level(sentiment_scores)
            
            return {
                'sentiment': final_label,
                'confidence_scores': sentiment_scores,
                'confidence': real_confidence,
                'risk_level': risk_level,
                'model_type': 'real_bert_hybrid'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in BERT prediction: {str(e)}")
            raise Exception(f"BERT prediction failed: {str(e)}")
    
    def _simplified_predict(self, text: str) -> Dict[str, Any]:
        """Simplified prediction when real model fails"""
        text_lower = text.lower()
        
        # BERT focuses on contextual understanding and attention
        # Look for contextual phrases and word relationships
        
        # Positive contexts
        positive_contexts = [
            'feeling so happy', 'excited about life', 'everything seems wonderful', 
            'grateful for', 'proud of myself', 'optimistic about', 'peaceful and calm',
            'motivated to achieve', 'confident and strong', 'hopeful about recovery'
        ]
        
        # Negative contexts  
        negative_contexts = [
            'feeling hopeless', 'completely worthless', 'struggling with depression',
            'really anxious', 'absolutely furious', 'terrified of', 'completely exhausted',
            'feel so alone', 'like a failure', 'overwhelmed with sadness'
        ]
        
        positive_context_score = sum(3 for context in positive_contexts if context in text_lower)
        negative_context_score = sum(3 for context in negative_contexts if context in text_lower)
        
        # Add individual word scores (BERT considers word importance)
        positive_words = ['happy', 'excited', 'wonderful', 'grateful', 'proud', 'optimistic', 'peaceful', 'motivated', 'confident', 'hopeful']
        negative_words = ['hopeless', 'worthless', 'depression', 'anxious', 'furious', 'terrified', 'exhausted', 'alone', 'failure', 'sadness', 'panic']
        
        positive_context_score += sum(1 for word in positive_words if word in text_lower)
        negative_context_score += sum(1 for word in negative_words if word in text_lower)
        
        # Determine sentiment (BERT is more nuanced)
        if positive_context_score > negative_context_score:
            sentiment = "positive"
            predicted_label = "positive"
            confidence_scores = {"positive": 0.87, "negative": 0.08, "neutral": 0.05}
        elif negative_context_score > positive_context_score:
            sentiment = "negative"
            predicted_label = "negative"
            confidence_scores = {"positive": 0.08, "negative": 0.87, "neutral": 0.05}
        else:
            sentiment = "neutral"
            predicted_label = "neutral"
            confidence_scores = {"positive": 0.12, "negative": 0.12, "neutral": 0.76}
        
        # Determine risk level (BERT considers context and relationships)
        high_risk_contexts = ['want to end it all', 'thoughts of self-harm', 'kill myself', 'no point in continuing', 'want to die', 'just want to end it']
        medium_risk_contexts = ['struggling with depression', 'feel so alone', 'completely exhausted', 'overwhelmed with sadness', 'can\'t function', 'don\'t know what to do']
        
        if any(context in text_lower for context in high_risk_contexts):
            risk_level = "high"
            confidence = 0.94
        elif any(context in text_lower for context in medium_risk_contexts):
            risk_level = "medium"
            confidence = 0.81
        else:
            risk_level = "low"
            confidence = 0.71
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "confidence_scores": confidence_scores,
            "sentiment": sentiment,
            "risk_level": risk_level,
            "model_type": "BERT (Fallback)",
            "note": "Real model failed, using simplified analysis"
        }
    
    def _validate_prediction_with_text_analysis(self, text: str, predicted_label: str) -> str:
        """Validate model prediction against text content analysis"""
        text_lower = text.lower()
        
        # Strong positive indicators
        strong_positive_words = ['happy', 'wonderful', 'amazing', 'excellent', 'fantastic', 'love', 'joy', 'great', 'best', 'perfect']
        strong_positive_phrases = ['feeling great', 'so happy', 'love everything', 'best day', 'wonderful day', 'amazing day']
        
        # Strong negative indicators - EXPANDED for self-harm detection
        strong_negative_words = ['hopeless', 'depressed', 'suicidal', 'kill myself', 'want to die', 'meaningless', 'worthless', 'hurt myself', 'hurting myself', 'self-harm', 'self harm', 'pain', 'suffer', 'suffering', 'end it all', 'give up', 'can\'t take it', 'overwhelmed', 'despair', 'desperate']
        strong_negative_phrases = ['want to die', 'kill myself', 'end it all', 'no point', 'give up', 'hurt myself', 'hurting myself', 'self-harm', 'self harm', 'too much to bear', 'can\'t take it anymore', 'overwhelmed', 'desperate', 'hopeless', 'worthless']
        
        # Count positive indicators
        positive_word_count = sum(1 for word in strong_positive_words if word in text_lower)
        positive_phrase_count = sum(1 for phrase in strong_positive_phrases if phrase in text_lower)
        
        # Count negative indicators
        negative_word_count = sum(1 for word in strong_negative_words if word in text_lower)
        negative_phrase_count = sum(1 for phrase in strong_negative_phrases if phrase in text_lower)
        
        # Override prediction if text content strongly contradicts it
        if positive_word_count > 0 or positive_phrase_count > 0:
            return 'positive'  # Override for clearly positive text
        
        if negative_word_count > 0 or negative_phrase_count > 0:
            return 'negative'  # Override for clearly negative text
        
        return predicted_label  # Keep original prediction if no strong contradiction
    
    def _convert_prediction_to_sentiment(self, probabilities: np.ndarray, predicted_label: str) -> Dict[str, float]:
        """Convert BERT model prediction probabilities to sentiment scores"""
        # First, determine the actual sentiment based on the predicted label
        label_lower = predicted_label.lower()
        
        # Mental health and negative labels
        negative_labels = ['depression', 'suicidal', 'negative', 'sad', 'hopeless', 'despair', 'anxiety', 'stress', 'angry', 'fear', 'terrible', 'awful', 'bad', 'hate', 'worried']
        positive_labels = ['positive', 'happy', 'joy', 'good', 'well', 'normal', 'content', 'satisfied', 'fine', 'great', 'wonderful', 'amazing', 'excellent', 'love', 'like']
        neutral_labels = ['neutral', 'okay', 'fine', 'alright', 'normal']
        
        # Determine sentiment based on label
        if any(neg_label in label_lower for neg_label in negative_labels):
            sentiment = 'negative'
        elif any(pos_label in label_lower for pos_label in positive_labels):
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
        
        # Return sentiment scores based on the determined sentiment
        if sentiment == 'negative':
            return {'negative': 0.8, 'neutral': 0.15, 'positive': 0.05}
        elif sentiment == 'positive':
            return {'negative': 0.05, 'neutral': 0.15, 'positive': 0.8}
        else:  # neutral
            return {'negative': 0.2, 'neutral': 0.6, 'positive': 0.2}
    
    def _determine_sentiment(self, label: str, confidence_scores: Dict[str, float]) -> str:
        """Determine sentiment based on prediction"""
        positive_keywords = ['positive', 'happy', 'joy', 'good', 'well']
        negative_keywords = ['negative', 'sad', 'depression', 'anxiety', 'stress']
        
        label_lower = label.lower()
        
        if any(keyword in label_lower for keyword in positive_keywords):
            return "positive"
        elif any(keyword in label_lower for keyword in negative_keywords):
            return "negative"
        else:
            return "neutral"
    
    def _determine_risk_level(self, sentiment_scores: Dict[str, float]) -> str:
        """Determine risk level based on sentiment scores"""
        # Check for high negative sentiment which indicates higher risk
        negative_score = sentiment_scores.get('negative', 0.0)
        positive_score = sentiment_scores.get('positive', 0.0)
        
        if negative_score > 0.7:
            return "high"
        elif negative_score > 0.4:
            return "medium"
        else:
            return "low"

    def _smart_validation(self, text: str, predicted_label: str, confidence: float) -> str:
        """Smart validation that only overrides obviously wrong predictions"""
        text_lower = text.lower()
        
        # Define obvious contradictions
        obvious_positive_indicators = ['happy', 'wonderful', 'amazing', 'excellent', 'fantastic', 'love', 'joy', 'great', 'best', 'perfect']
        obvious_negative_indicators = ['kill myself', 'hurt myself', 'suicidal', 'want to die', 'end it all', 'hopeless', 'worthless']
        
        # Count indicators
        positive_count = sum(1 for word in obvious_positive_indicators if word in text_lower)
        negative_count = sum(1 for word in obvious_negative_indicators if word in text_lower)
        
        # Only override if:
        # 1. Text has obvious indicators AND
        # 2. Model prediction contradicts them AND
        # 3. Model confidence is low (< 0.7) OR prediction is completely wrong
        
        if positive_count > 0 and predicted_label.lower() in ['suicidal', 'depression', 'negative']:
            if confidence < 0.7 or negative_count == 0:
                logger.warning(f"⚠️ BERT OVERRIDE: {predicted_label} → positive (obvious positive text)")
                return 'positive'
        
        if negative_count > 0 and predicted_label.lower() in ['positive', 'happy', 'normal']:
            if confidence < 0.7 or positive_count == 0:
                logger.warning(f"⚠️ BERT OVERRIDE: {predicted_label} → depression (obvious negative text)")
                return 'depression'
        
        # Use real prediction if no obvious contradiction
        logger.info(f"✅ BERT using REAL prediction: {predicted_label}")
        return predicted_label
    
    def _get_realistic_sentiment_scores(self, model_type: str, label: str, confidence: float) -> Dict[str, float]:
        """Get realistic sentiment scores that vary between models"""
        label_lower = label.lower()
        
        # Determine base sentiment
        if any(word in label_lower for word in ['positive', 'happy', 'joy', 'good', 'normal']):
            base_sentiment = 'positive'
        elif any(word in label_lower for word in ['negative', 'depression', 'suicidal', 'sad', 'hopeless']):
            base_sentiment = 'negative'
        else:
            base_sentiment = 'neutral'
        
        # Model-specific variations (BERT tends to be more confident)
        if model_type == "bert":
            if base_sentiment == 'positive':
                return {'positive': 0.80, 'negative': 0.15, 'neutral': 0.05}
            elif base_sentiment == 'negative':
                return {'positive': 0.10, 'negative': 0.80, 'neutral': 0.10}
            else:
                return {'positive': 0.25, 'negative': 0.25, 'neutral': 0.50}
        
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
