"""
Data Preprocessing Service

This service handles text preprocessing for mental health analysis.
It cleans, normalizes, and prepares text data for AI model processing.

Phase 2: Data Preprocessing
- Text cleaning and normalization
- Feature extraction
- Sentiment preprocessing
- Data validation and filtering
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import asyncio
import functools

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """
    Advanced text preprocessing for mental health analysis
    
    This class provides comprehensive text preprocessing including:
    1. Text cleaning and normalization
    2. Tokenization and lemmatization
    3. Feature extraction
    4. Sentiment analysis preparation
    5. Mental health specific preprocessing
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Mental health related keywords for feature extraction
        self.mental_health_keywords = {
            'depression': ['depressed', 'sad', 'hopeless', 'worthless', 'empty', 'down', 'blue', 'melancholy'],
            'anxiety': ['anxious', 'worried', 'nervous', 'stressed', 'panic', 'fear', 'tense', 'uneasy'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed', 'frustrated', 'hostile'],
            'joy': ['happy', 'joyful', 'excited', 'cheerful', 'elated', 'glad', 'delighted', 'pleased'],
            'fear': ['scared', 'afraid', 'terrified', 'frightened', 'intimidated', 'alarmed', 'threatened'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated', 'appalled']
        }
        
        # Negation words that can change sentiment
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'none',
            'hardly', 'scarcely', 'barely', 'seldom', 'rarely', "n't", 'cannot', 'cant'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 1.8,
            'really': 1.3, 'quite': 1.2, 'rather': 1.1, 'somewhat': 0.8,
            'slightly': 0.6, 'barely': 0.4, 'hardly': 0.3
        }
    
    async def preprocess_text(self, text: str, advanced: bool = True) -> Dict[str, Any]:
        """
        Comprehensive text preprocessing
        
        Args:
            text: Input text to preprocess
            advanced: Whether to include advanced features
            
        Returns:
            Dict: Preprocessed text and extracted features
        """
        try:
            if not text or not text.strip():
                return self._empty_result()
            
            # Run preprocessing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Basic preprocessing
            cleaned_text = await loop.run_in_executor(
                None, 
                functools.partial(self._clean_text, text)
            )
            
            if not cleaned_text:
                return self._empty_result()
            
            # Tokenization and basic features
            tokens = await loop.run_in_executor(
                None,
                functools.partial(self._tokenize_and_process, cleaned_text)
            )
            
            # Advanced features if requested
            features = {}
            if advanced:
                features = await loop.run_in_executor(
                    None,
                    functools.partial(self._extract_advanced_features, text, cleaned_text, tokens)
                )
            
            # Basic statistics
            stats = self._calculate_text_stats(text, tokens)
            
            result = {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'tokens': tokens,
                'statistics': stats,
                'features': features,
                'is_valid': len(tokens) > 2 and stats['word_count'] > 3
            }
            
            logger.debug(f"✅ Preprocessed text: {stats['word_count']} words, {stats['sentence_count']} sentences")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing text: {str(e)}")
            return self._empty_result(error=str(e))
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove Twitter handles and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        
        # Clean up punctuation
        text = re.sub(r'\.{2,}', '.', text)  # Multiple periods
        text = re.sub(r'\!{2,}', '!', text)  # Multiple exclamations
        text = re.sub(r'\?{2,}', '?', text)  # Multiple questions
        
        return text.strip()
    
    def _tokenize_and_process(self, text: str) -> List[str]:
        """
        Tokenize and process text
        
        Args:
            text: Cleaned text
            
        Returns:
            List[str]: Processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words 
            and token not in string.punctuation
            and len(token) > 2
            and token.isalpha()
        ]
        
        return tokens
    
    def _extract_advanced_features(self, original_text: str, cleaned_text: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Extract advanced text features
        
        Args:
            original_text: Original text
            cleaned_text: Cleaned text
            tokens: Processed tokens
            
        Returns:
            Dict: Advanced features
        """
        features = {}
        
        # Mental health keyword analysis
        features['mental_health_keywords'] = self._analyze_mental_health_keywords(tokens)
        
        # Sentiment polarity using TextBlob
        blob = TextBlob(cleaned_text)
        features['textblob_sentiment'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # Negation analysis
        features['negation_count'] = self._count_negations(original_text)
        
        # Intensity analysis
        features['intensity_score'] = self._calculate_intensity(tokens)
        
        # Emotional indicators
        features['emotional_indicators'] = self._analyze_emotional_indicators(original_text)
        
        # Linguistic features
        features['linguistic'] = self._extract_linguistic_features(original_text, tokens)
        
        return features
    
    def _analyze_mental_health_keywords(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Analyze mental health related keywords
        
        Args:
            tokens: Text tokens
            
        Returns:
            Dict: Mental health keyword analysis
        """
        keyword_counts = {}
        total_keywords = 0
        
        for category, keywords in self.mental_health_keywords.items():
            count = sum(1 for token in tokens if token in keywords)
            keyword_counts[category] = count
            total_keywords += count
        
        return {
            'counts': keyword_counts,
            'total': total_keywords,
            'dominant_category': max(keyword_counts, key=keyword_counts.get) if total_keywords > 0 else None
        }
    
    def _count_negations(self, text: str) -> int:
        """
        Count negation words in text
        
        Args:
            text: Input text
            
        Returns:
            int: Number of negations
        """
        words = text.lower().split()
        return sum(1 for word in words if any(neg in word for neg in self.negation_words))
    
    def _calculate_intensity(self, tokens: List[str]) -> float:
        """
        Calculate text intensity based on intensifier words
        
        Args:
            tokens: Text tokens
            
        Returns:
            float: Intensity score
        """
        intensity_sum = 0.0
        count = 0
        
        for token in tokens:
            if token in self.intensifiers:
                intensity_sum += self.intensifiers[token]
                count += 1
        
        return intensity_sum / count if count > 0 else 1.0
    
    def _analyze_emotional_indicators(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotional indicators in text
        
        Args:
            text: Input text
            
        Returns:
            Dict: Emotional indicators
        """
        # Count punctuation that indicates emotion
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_count = sum(1 for c in text if c.isupper())
        
        # Detect emotional expressions
        emotional_expressions = re.findall(r'[hH]a+h+a+|[lL]o+l+|[oO]mg+|[wW]ow+', text)
        
        return {
            'exclamation_marks': exclamation_count,
            'question_marks': question_count,
            'capital_letters': caps_count,
            'emotional_expressions': len(emotional_expressions),
            'text_length': len(text)
        }
    
    def _extract_linguistic_features(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Extract linguistic features
        
        Args:
            text: Original text
            tokens: Processed tokens
            
        Returns:
            Dict: Linguistic features
        """
        sentences = sent_tokenize(text)
        
        return {
            'avg_word_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
            'avg_sentence_length': len(text.split()) / len(sentences) if sentences else 0,
            'type_token_ratio': len(set(tokens)) / len(tokens) if tokens else 0,
            'sentence_count': len(sentences),
            'unique_words': len(set(tokens))
        }
    
    def _calculate_text_stats(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Calculate basic text statistics
        
        Args:
            text: Original text
            tokens: Processed tokens
            
        Returns:
            Dict: Text statistics
        """
        sentences = sent_tokenize(text)
        words = text.split()
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'token_count': len(tokens),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0
        }
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Return empty preprocessing result
        
        Args:
            error: Optional error message
            
        Returns:
            Dict: Empty result
        """
        return {
            'original_text': '',
            'cleaned_text': '',
            'tokens': [],
            'statistics': {},
            'features': {},
            'is_valid': False,
            'error': error
        }
    
    async def preprocess_batch(self, texts: List[str], advanced: bool = True) -> List[Dict[str, Any]]:
        """
        Preprocess multiple texts in batch
        
        Args:
            texts: List of texts to preprocess
            advanced: Whether to include advanced features
            
        Returns:
            List[Dict]: List of preprocessing results
        """
        try:
            logger.info(f"🔄 Preprocessing batch of {len(texts)} texts")
            
            # Process all texts concurrently
            tasks = [self.preprocess_text(text, advanced) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Error processing text {i}: {str(result)}")
                    processed_results.append(self._empty_result(error=str(result)))
                else:
                    processed_results.append(result)
            
            valid_count = sum(1 for r in processed_results if r.get('is_valid', False))
            logger.info(f"✅ Batch preprocessing complete: {valid_count}/{len(texts)} valid texts")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Error in batch preprocessing: {str(e)}")
            return [self._empty_result(error=str(e)) for _ in texts]


# Utility functions for preprocessing
def clean_social_media_text(text: str) -> str:
    """
    Quick social media text cleaning
    
    Args:
        text: Raw social media text
        
    Returns:
        str: Cleaned text
    """
    # Remove RT (retweet) prefix
    text = re.sub(r'^RT\s+', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'[@#]\w+', '', text)
    
    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


async def test_preprocessing():
    """
    Test function for preprocessing service
    """
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "I'm feeling really depressed today. Nothing seems to work out for me. 😔",
        "OMG! This is absolutely amazing!!! I'm so excited and happy! 🎉",
        "I don't know what to do anymore. Everything is just too overwhelming and scary.",
        "Having a great day with friends. Love spending time with people who care! ❤️"
    ]
    
    print("🔄 Testing text preprocessing...")
    
    for i, text in enumerate(test_texts, 1):
        result = await preprocessor.preprocess_text(text)
        print(f"\n--- Test {i} ---")
        print(f"Original: {text}")
        print(f"Cleaned: {result['cleaned_text']}")
        print(f"Tokens: {len(result['tokens'])}")
        print(f"Valid: {result['is_valid']}")
        
        if result['features']:
            sentiment = result['features']['textblob_sentiment']
            keywords = result['features']['mental_health_keywords']
            print(f"Sentiment: {sentiment['polarity']:.2f}")
            print(f"Keywords: {keywords['total']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_preprocessing())
