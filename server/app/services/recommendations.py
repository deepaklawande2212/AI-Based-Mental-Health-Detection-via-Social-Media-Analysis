"""
Recommendations Service

This service provides intelligent recommendations based on mental health analysis results.
It considers sentiment, emotions, risk levels, and specific factors to generate personalized advice.
"""

from typing import List, Dict, Any
from loguru import logger


class RecommendationEngine:
    """Intelligent recommendation engine for mental health analysis"""
    
    def __init__(self):
        # Risk-based recommendations
        self.risk_recommendations = {
            'low': [
                "Continue monitoring your mental health regularly",
                "Maintain healthy lifestyle habits including exercise and sleep",
                "Practice stress management techniques like meditation or deep breathing",
                "Stay connected with friends and family for social support",
                "Consider journaling to track your emotional well-being"
            ],
            'medium': [
                "Consider speaking with a mental health professional for support",
                "Practice self-care activities that bring you joy and relaxation",
                "Establish a regular sleep schedule and prioritize rest",
                "Limit exposure to negative news and social media",
                "Engage in physical activity to boost mood and reduce stress",
                "Consider joining a support group or talking to trusted friends"
            ],
            'high': [
                "Seek immediate professional mental health support",
                "Contact a crisis helpline if you're having thoughts of self-harm",
                "Reach out to trusted friends, family, or healthcare providers",
                "Consider medication evaluation with a psychiatrist if recommended",
                "Create a safety plan with your mental health professional",
                "Remove access to harmful items and ensure your environment is safe"
            ]
        }
        
        # Emotion-specific recommendations
        self.emotion_recommendations = {
            'joy': [
                "Continue activities that bring you happiness and fulfillment",
                "Share your positive energy with others who may need support",
                "Document what's working well in your life for future reference",
                "Use your positive mood to set and work toward personal goals"
            ],
            'sadness': [
                "Allow yourself to feel and process your emotions without judgment",
                "Practice self-compassion and be kind to yourself",
                "Engage in activities that typically bring you comfort",
                "Consider talking to a therapist about your feelings",
                "Try gentle physical activities like walking or yoga"
            ],
            'anger': [
                "Practice deep breathing or counting to 10 before reacting",
                "Identify triggers and develop healthy coping strategies",
                "Consider anger management techniques or therapy",
                "Express your feelings through writing or creative outlets",
                "Take time to cool down before addressing conflicts"
            ],
            'fear': [
                "Practice grounding techniques to stay present",
                "Identify specific fears and challenge irrational thoughts",
                "Consider anxiety management strategies or therapy",
                "Create a calming routine for stressful situations",
                "Limit caffeine and ensure adequate sleep"
            ],
            'surprise': [
                "Take time to process unexpected events or information",
                "Seek clarification if something is unclear or confusing",
                "Allow yourself to adjust to new circumstances",
                "Consider how this new information affects your plans"
            ],
            'disgust': [
                "Identify what's causing the feeling of disgust",
                "Set healthy boundaries with people or situations that trigger this emotion",
                "Practice self-care to restore your sense of well-being",
                "Consider if this emotion is protecting you from something harmful"
            ]
        }
        
        # Sentiment-based recommendations
        self.sentiment_recommendations = {
            'positive': [
                "Maintain the positive mindset and activities that are working well",
                "Share your positive outlook with others who may benefit",
                "Use this positive energy to work toward personal goals",
                "Document what's contributing to your positive state"
            ],
            'negative': [
                "Consider speaking with a mental health professional",
                "Practice self-care and stress management techniques",
                "Identify and challenge negative thought patterns",
                "Engage in activities that typically improve your mood",
                "Reach out to trusted friends or family for support"
            ],
            'neutral': [
                "Monitor your emotional state for any changes",
                "Consider what might help you feel more engaged or positive",
                "Practice mindfulness to become more aware of your feelings",
                "Try new activities to discover what brings you joy"
            ]
        }
        
        # Crisis recommendations
        self.crisis_recommendations = [
            "Call emergency services (911) immediately if you're in immediate danger",
            "Contact the National Suicide Prevention Lifeline at 988",
            "Text HOME to 741741 to reach the Crisis Text Line",
            "Go to the nearest emergency room for immediate help",
            "Remove access to any means of self-harm",
            "Stay with a trusted person until you can get professional help"
        ]

    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate comprehensive recommendations based on analysis results
        
        Args:
            analysis_results: Dictionary containing sentiment, emotions, risk, and other analysis data
            
        Returns:
            List of personalized recommendations
        """
        recommendations = []
        
        try:
            # Get risk level
            risk_level = analysis_results.get('risk_level', 'low').lower()
            
            # Get dominant emotion
            emotions = analysis_results.get('emotions', {})
            dominant_emotion = emotions.get('dominant_emotion', 'neutral')
            
            # Get sentiment
            sentiment = analysis_results.get('sentiment', {})
            overall_sentiment = sentiment.get('overall', 'neutral')
            
            # Add risk-based recommendations
            if risk_level in self.risk_recommendations:
                recommendations.extend(self.risk_recommendations[risk_level])
            
            # Add emotion-specific recommendations
            if dominant_emotion in self.emotion_recommendations:
                emotion_recs = self.emotion_recommendations[dominant_emotion]
                # Add 2-3 emotion-specific recommendations
                recommendations.extend(emotion_recs[:3])
            
            # Add sentiment-based recommendations
            if overall_sentiment in self.sentiment_recommendations:
                sentiment_recs = self.sentiment_recommendations[overall_sentiment]
                # Add 1-2 sentiment-based recommendations
                recommendations.extend(sentiment_recs[:2])
            
            # Add crisis recommendations for high-risk cases
            if risk_level == 'high':
                # Check for specific crisis indicators
                text_content = analysis_results.get('text_content', '').lower()
                crisis_indicators = ['suicide', 'kill myself', 'end it all', 'want to die', 'better off dead']
                
                if any(indicator in text_content for indicator in crisis_indicators):
                    recommendations = self.crisis_recommendations + recommendations[:3]
            
            # Add general mental health recommendations
            general_recs = [
                "Consider regular mental health check-ups",
                "Maintain a balanced diet and regular exercise routine",
                "Practice good sleep hygiene and aim for 7-9 hours of sleep",
                "Limit alcohol and avoid recreational drugs",
                "Stay connected with supportive friends and family"
            ]
            
            # Add 1-2 general recommendations if we have room
            if len(recommendations) < 8:
                recommendations.extend(general_recs[:2])
            
            # Remove duplicates and limit to 8-10 recommendations
            unique_recommendations = list(dict.fromkeys(recommendations))
            return unique_recommendations[:10]
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {str(e)}")
            # Return basic recommendations as fallback
            return [
                "Consider speaking with a mental health professional",
                "Practice self-care and stress management",
                "Maintain healthy lifestyle habits",
                "Stay connected with supportive people"
            ]


# Global recommendation engine instance
recommendation_engine = RecommendationEngine()


def generate_analysis_recommendations(analysis_data: Dict[str, Any]) -> List[str]:
    """
    Generate recommendations for mental health analysis results
    
    Args:
        analysis_data: Dictionary containing analysis results
        
    Returns:
        List of personalized recommendations
    """
    return recommendation_engine.generate_recommendations(analysis_data) 