"""
FinBERT sentiment analysis for financial text.
"""
import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinBERTSentiment:
    """
    FinBERT sentiment analysis for financial text.
    """
    
    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Initialize the FinBERT sentiment analyzer.
        
        Args:
            model_name: Name of the pre-trained model
            device: Device to use for inference (None for auto-detection)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        logger.info(f"Initializing FinBERT sentiment analyzer with model {model_name}")
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Set batch size and max length
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Get label mapping
        if hasattr(self.model.config, "id2label"):
            self.id2label = self.model.config.id2label
        else:
            # Default FinBERT labels
            self.id2label = {0: "positive", 1: "neutral", 2: "negative"}
    
    def analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment scores
        """
        if not texts:
            return []
        
        logger.info(f"Analyzing sentiment of {len(texts)} texts")
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_results = self._analyze_batch(batch_texts)
            results.extend(batch_results)
        
        return results
    
    def _analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment scores
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Create results
        batch_results = []
        for prob in probs:
            result = {self.id2label[i]: float(p) for i, p in enumerate(prob)}
            batch_results.append(result)
        
        return batch_results
    
    def analyze_with_text(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of texts and include the text in the results.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with text and sentiment scores
        """
        sentiment_results = self.analyze(texts)
        
        results = []
        for text, sentiment in zip(texts, sentiment_results):
            result = {"text": text}
            result.update(sentiment)
            results.append(result)
        
        return results
    
    def get_sentiment_label(self, text: str) -> str:
        """
        Get the sentiment label (positive, neutral, negative) for a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment label
        """
        result = self.analyze([text])[0]
        return max(result.items(), key=lambda x: x[1])[0]
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get a single sentiment score for a text.
        
        The score is in the range [-1, 1], where:
        - 1 is very positive
        - 0 is neutral
        - -1 is very negative
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score
        """
        result = self.analyze([text])[0]
        
        # Calculate a single score
        # positive - negative, with neutral reducing the magnitude
        score = result.get("positive", 0) - result.get("negative", 0)
        
        # Reduce magnitude based on neutral score
        neutral = result.get("neutral", 0)
        score *= (1 - neutral)
        
        return score

def batch_analyze_texts(
    texts: List[str],
    model_name: str = "yiyanghkust/finbert-tone",
    batch_size: int = 16,
    max_length: int = 512
) -> List[Dict[str, float]]:
    """
    Analyze sentiment of a list of texts.
    
    Args:
        texts: List of texts to analyze
        model_name: Name of the pre-trained model
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        
    Returns:
        List of dictionaries with sentiment scores
    """
    analyzer = FinBERTSentiment(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length
    )
    
    return analyzer.analyze(texts)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze sentiment of financial texts")
    parser.add_argument("--texts", type=str, nargs="+", help="Texts to analyze")
    parser.add_argument("--model-name", type=str, default="yiyanghkust/finbert-tone", help="Name of the pre-trained model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    if args.texts:
        analyzer = FinBERTSentiment(
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        results = analyzer.analyze_with_text(args.texts)
        
        for result in results:
            text = result.pop("text")
            sentiment_label = max(result.items(), key=lambda x: x[1])[0]
            sentiment_score = result.get("positive", 0) - result.get("negative", 0)
            
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment_label} (score: {sentiment_score:.2f})")
            print(f"Probabilities: {result}")
            print()
