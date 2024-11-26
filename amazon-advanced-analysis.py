import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from typing import Optional, List, Dict, Tuple, Union
import logging
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
import requests
from scipy.sparse.linalg import svds
import json
import warnings
warnings.filterwarnings('ignore')

class AmazonAdvancedAnalyzer:
    def __init__(self, data_path: str):
        """Initialize the advanced analyzer with expanded capabilities."""
        self.logger = self._setup_logging()
        self.df = self._load_and_preprocess_data(data_path)
        self.le = LabelEncoder()
        self.sia = SentimentIntensityAnalyzer()
        self.price_history = {}
        self.setup_nlp()
        self.setup_image_model()
        
    def setup_nlp(self):
        """Set up NLP components."""
        nltk.download('vader_lexicon')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        
    def setup_image_model(self):
        """Set up image processing model."""
        self.image_model = ResNet50(weights='imagenet', include_top=False)

    class PriceTracker:
        def __init__(self):
            self.price_history = {}
            
        def update_price(self, product_id: str, price: float):
            """Update price history for a product."""
            if product_id not in self.price_history:
                self.price_history[product_id] = []
            
            self.price_history[product_id].append({
                'timestamp': datetime.now(),
                'price': price
            })
            
        def get_price_trends(self, product_id: str, days: int = 30) -> pd.DataFrame:
            """Get price trends for a specific product."""
            if product_id not in self.price_history:
                return pd.DataFrame()
                
            history = self.price_history[product_id]
            df_history = pd.DataFrame(history)
            
            # Calculate various metrics
            df_history['price_change'] = df_history['price'].diff()
            df_history['price_change_pct'] = df_history['price'].pct_change() * 100
            
            return df_history

    class CollaborativeFilter:
        def __init__(self, ratings_matrix: pd.DataFrame):
            """Initialize collaborative filtering system."""
            self.ratings_matrix = ratings_matrix
            self.user_similarity = None
            self.item_similarity = None
            
        def train(self, n_factors: int = 50):
            """Train the collaborative filtering model."""
            # Convert ratings matrix to numpy array
            R = self.ratings_matrix.values
            
            # Normalize the ratings
            user_ratings_mean = np.mean(R, axis=1)
            R_demeaned = R - user_ratings_mean.reshape(-1, 1)
            
            # Perform SVD
            U, sigma, Vt = svds(R_demeaned, k=n_factors)
            
            # Convert sigma to diagonal matrix
            sigma = np.diag(sigma)
            
            # Calculate predicted ratings
            self.predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
            
        def get_recommendations(self, user_id: int, n_items: int = 5) -> List[int]:
            """Get recommendations for a specific user."""
            user_predictions = self.predicted_ratings[user_id]
            
            # Get items the user hasn't rated
            unrated_items = np.where(self.ratings_matrix.iloc[user_id].isna())[0]
            
            # Sort predictions and get top n
            recommendations = unrated_items[np.argsort(user_predictions[unrated_items])[-n_items:]]
            
            return recommendations.tolist()

    class SeasonalAnalyzer:
        def __init__(self, sales_data: pd.DataFrame):
            """Initialize seasonal analysis system."""
            self.sales_data = sales_data
            
        def analyze_seasonal_trends(self) -> Dict[str, pd.DataFrame]:
            """Analyze seasonal trends in sales data."""
            # Add datetime components
            self.sales_data['month'] = pd.to_datetime(self.sales_data['date']).dt.month
            self.sales_data['quarter'] = pd.to_datetime(self.sales_data['date']).dt.quarter
            
            # Calculate seasonal metrics
            seasonal_metrics = {
                'monthly_sales': self.sales_data.groupby('month')['sales'].mean(),
                'quarterly_sales': self.sales_data.groupby('quarter')['sales'].mean(),
                'monthly_trends': self._calculate_monthly_trends()
            }
            
            return seasonal_metrics
            
        def _calculate_monthly_trends(self) -> pd.DataFrame:
            """Calculate detailed monthly trends."""
            monthly_data = self.sales_data.groupby('month').agg({
                'sales': ['mean', 'std', 'count'],
                'discount_percentage': 'mean'
            })
            
            return monthly_data

    class ABTester:
        def __init__(self):
            """Initialize A/B testing system."""
            self.experiments = {}
            self.results = {}
            
        def create_experiment(self, name: str, variants: List[str]):
            """Create a new A/B test experiment."""
            self.experiments[name] = {
                'variants': variants,
                'data': {variant: [] for variant in variants}
            }
            
        def record_result(self, experiment: str, variant: str, success: bool):
            """Record the result of a variant test."""
            if experiment in self.experiments:
                if variant in self.experiments[experiment]['variants']:
                    self.experiments[experiment]['data'][variant].append(success)
                    
        def analyze_results(self, experiment: str) -> Dict[str, float]:
            """Analyze the results of an experiment."""
            if experiment not in self.experiments:
                return {}
                
            results = {}
            for variant, data in self.experiments[experiment]['data'].items():
                if data:
                    success_rate = sum(data) / len(data)
                    results[variant] = success_rate
                    
            return results

    class ImageRecommender:
        def __init__(self, model: tf.keras.Model):
            """Initialize image-based recommendation system."""
            self.model = model
            self.image_features = {}
            
        def extract_features(self, image_path: str) -> np.ndarray:
            """Extract features from an image."""
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            features = self.model.predict(x)
            return features.flatten()
            
        def add_product_image(self, product_id: str, image_path: str):
            """Add a product image to the recommendation system."""
            features = self.extract_features(image_path)
            self.image_features[product_id] = features
            
        def get_similar_products(self, product_id: str, n_similar: int = 5) -> List[str]:
            """Get visually similar products."""
            if product_id not in self.image_features:
                return []
                
            target_features = self.image_features[product_id]
            similarities = {}
            
            for pid, features in self.image_features.items():
                if pid != product_id:
                    similarity = cosine_similarity(
                        target_features.reshape(1, -1),
                        features.reshape(1, -1)
                    )[0][0]
                    similarities[pid] = similarity
                    
            similar_products = sorted(similarities.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:n_similar]
                                   
            return [pid for pid, _ in similar_products]

    class NLPAnalyzer:
        def __init__(self):
            """Initialize NLP analysis system."""
            self.sia = SentimentIntensityAnalyzer()
            
        def analyze_review(self, review: str) -> Dict[str, Union[float, List[str]]]:
            """Perform detailed analysis of a review."""
            # Sentiment analysis
            sentiment = self.sia.polarity_scores(review)
            
            # Extract key phrases
            tokens = word_tokenize(review.lower())
            key_phrases = [token for token in tokens if token not in stopwords.words('english')]
            
            # Identify product aspects
            aspects = self._extract_aspects(review)
            
            return {
                'sentiment': sentiment,
                'key_phrases': key_phrases,
                'aspects': aspects
            }
            
        def _extract_aspects(self, review: str) -> List[str]:
            """Extract product aspects from review."""
            # Simplified aspect extraction
            aspects = []
            sentences = nltk.sent_tokenize(review)
            
            for sentence in sentences:
                tokens = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(tokens)
                
                # Extract nouns as potential aspects
                aspects.extend([word for word, pos in tagged 
                              if pos in ['NN', 'NNS']])
                
            return aspects

    def generate_comprehensive_recommendations(self, user_id: int) -> Dict[str, List[str]]:
        """Generate recommendations using multiple methods."""
        recommendations = {
            'collaborative': self._get_collaborative_recommendations(user_id),
            'content_based': self._get_content_based_recommendations(user_id),
            'image_based': self._get_image_based_recommendations(user_id),
            'seasonal': self._get_seasonal_recommendations(user_id)
        }
        
        # Run A/B test
        self.ab_tester.record_result('recommendation_method', 
                                   'comprehensive', 
                                   True)  # Record success/failure based on user interaction
                                   
        return recommendations

    def analyze_long_term_trends(self) -> Dict[str, pd.DataFrame]:
        """Analyze long-term trends in the data."""
        trends = {
            'price_trends': self._analyze_price_trends(),
            'rating_trends': self._analyze_rating_trends(),
            'category_trends': self._analyze_category_trends()
        }
        
        return trends

    def generate_insights_report(self) -> Dict[str, str]:
        """Generate comprehensive insights report."""
        insights = {
            'price_insights': self._generate_price_insights(),
            'rating_insights': self._generate_rating_insights(),
            'seasonal_insights': self._generate_seasonal_insights(),
            'nlp_insights': self._generate_nlp_insights()
        }
        
        return insights

# Example usage:
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = AmazonAdvancedAnalyzer("amazon.csv")
    
    # Set up A/B testing
    analyzer.ab_tester.create_experiment(
        'recommendation_method',
        ['collaborative', 'content_based', 'comprehensive']
    )
    
    # Generate recommendations for a user
    user_id = 893
    recommendations = analyzer.generate_comprehensive_recommendations(user_id)
    
    # Analyze trends
    trends = analyzer.analyze_long_term_trends()
    
    # Generate insights
    insights = analyzer.generate_insights_report()
    
    # Print results
    print("Recommendations:", recommendations)
    print("\nTrends:", trends)
    print("\nInsights:", insights)
