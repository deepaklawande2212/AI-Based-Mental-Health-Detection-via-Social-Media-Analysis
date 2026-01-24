"""
Risk Assessment Service

This module provides risk assessment functionality for text analysis.
"""

from typing import Dict, Any, List
from loguru import logger
import asyncio


class RiskAssessor:
    """Risk assessment service"""
    
    def __init__(self):
        """Initialize risk assessor"""
        self.name = "risk_assessor"
        
        # Risk indicators
        self.risk_indicators = {
            
            "suicidal": [
                "suicide", "kill myself", "end it all", "want to die", "better off dead",
                "no point living", "self-harm", "hurt myself", "end my life", "take my life"
            ],
            "depression": [
                "depression", "depressed", "hopeless", "worthless", "useless", "no hope",
                "give up", "can't go on", "life is meaningless", "no reason to live"
            ],
            "anxiety": [
                "panic", "anxiety", "anxious", "overwhelming", "can't breathe",
                "heart racing", "panic attack", "constant worry", "paralyzing fear"
            ],
            "anger": [
                "violent", "rage", "furious", "hate everyone", "want to hurt",
                "explode", "lose control", "violent thoughts"
            ]
        }
    
    async def assess_risk(self, text: str, prediction: str) -> Dict[str, Any]:
        """
        Assess risk level based on text and prediction
        
        Args:
            text: Text to analyze
            prediction: Model prediction
            
        Returns:
            Dict containing risk assessment results
        """
        try:
            # Simulate async processing
            await asyncio.sleep(0.1)
            
            text_lower = text.lower()
            prediction_lower = prediction.lower() if prediction else ""
            
            # Calculate risk score
            risk_score = 0.0
            risk_factors = []
            
            # Check for risk indicators in text
            for risk_type, indicators in self.risk_indicators.items():
                for indicator in indicators:
                    if indicator in text_lower:
                        risk_score += 0.3
                        risk_factors.append(f"{risk_type}: {indicator}")
            
            # Check prediction for high-risk categories
            high_risk_predictions = ["suicidal", "depression", "bipolar"]
            if any(risk_pred in prediction_lower for risk_pred in high_risk_predictions):
                risk_score += 0.4
                risk_factors.append(f"model_prediction: {prediction}")
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "high"
                recommendations = [
                    "Immediate professional help recommended",
                    "Contact mental health crisis line",
                    "Consider emergency services if in immediate danger"
                ]
            elif risk_score >= 0.4:
                risk_level = "medium"
                recommendations = [
                    "Professional counseling recommended",
                    "Consider talking to a mental health professional",
                    "Monitor symptoms and seek help if they worsen"
                ]
            else:
                risk_level = "low"
                recommendations = [
                    "Continue monitoring mental health",
                    "Consider preventive mental health care",
                    "Maintain healthy coping mechanisms"
                ]
            
            # Calculate confidence based on number of indicators
            confidence = min(0.95, 0.5 + (len(risk_factors) * 0.1))
            
            return {
                "level": risk_level,
                "score": round(risk_score, 3),
                "factors": risk_factors,
                "recommendations": recommendations,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in risk assessment: {str(e)}")
            return {
                "level": "low",
                "score": 0.0,
                "factors": [],
                "recommendations": ["Unable to assess risk - please consult a professional"],
                "confidence": 0.0
            } 