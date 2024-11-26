import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Dict, Tuple
import logging

class AmazonProductAnalyzer:
    def __init__(self, data_path: str):
        """Initialize the analyzer with data path."""
        self.logger = self._setup_logging()
        self.df = self._load_and_preprocess_data(data_path)
        self.le = LabelEncoder()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        self.logger.info("Loading data...")
        df = pd.read_csv(data_path)
        
        # Clean price columns
        for col in ['discounted_price', 'actual_price']:
            df[col] = df[col].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
        
        # Clean discount percentage
        df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%','').astype(float)/100
        
        # Clean rating columns
        df = df[~df['rating'].astype(str).str.contains('\\|')]
        df['rating'] = df['rating'].astype(float)
        df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(float)
        
        # Add weighted rating
        df['rating_weighted'] = df['rating'] * df['rating_count']
        
        # Extract categories
        df['sub_category'] = df['category'].astype(str).str.split('|').str[-1]
        df['main_category'] = df['category'].astype(str).str.split('|').str[0]
        
        # Encode user IDs
        df['user_id_encoded'] = self.le.fit_transform(df['user_id'])
        
        self.logger.info(f"Loaded {len(df)} records successfully")
        return df

    def analyze_categories(self, top_n: int = 30) -> Dict[str, pd.DataFrame]:
        """Analyze product categories distribution."""
        self.logger.info("Analyzing categories...")
        
        results = {}
        
        # Main categories analysis
        main_cats = self.df['main_category'].value_counts()[:top_n]
        results['main_categories'] = pd.DataFrame({
            'Category': main_cats.index,
            'Count': main_cats.values
        })
        
        # Sub categories analysis
        sub_cats = self.df['sub_category'].value_counts()[:top_n]
        results['sub_categories'] = pd.DataFrame({
            'Category': sub_cats.index,
            'Count': sub_cats.values
        })
        
        return results

    def analyze_ratings(self) -> Dict[str, pd.DataFrame]:
        """Analyze product ratings."""
        self.logger.info("Analyzing ratings...")
        
        results = {}
        
        # Rating distribution
        bins = [0, 1, 2, 3, 4, 5]
        self.df['rating_cluster'] = pd.cut(self.df['rating'], bins=bins, 
                                         labels=['0-1', '1-2', '2-3', '3-4', '4-5'])
        
        results['rating_distribution'] = self.df['rating_cluster'].value_counts().reset_index()
        
        # Average rating by category
        results['category_ratings'] = self.df.groupby('main_category')['rating'].agg(
            ['mean', 'count', 'std']).round(2).reset_index()
        
        return results

    def analyze_pricing(self) -> Dict[str, pd.DataFrame]:
        """Analyze pricing and discounts."""
        self.logger.info("Analyzing pricing...")
        
        results = {}
        
        # Discount analysis by category
        results['category_discounts'] = self.df.groupby('main_category').agg({
            'discount_percentage': ['mean', 'median', 'std'],
            'discounted_price': ['mean', 'median', 'std']
        }).round(3)
        
        # Price ranges
        results['price_ranges'] = pd.qcut(self.df['discounted_price'], 
                                        q=5).value_counts().reset_index()
        
        return results

    def generate_recommendations(self, user_id: int, n_recommendations: int = 5) -> Optional[pd.DataFrame]:
        """Generate product recommendations for a user."""
        self.logger.info(f"Generating recommendations for user {user_id}...")
        
        try:
            # Prepare text features
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(self.df['about_product'].fillna(''))
            
            # Get user history
            user_history = self.df[self.df['user_id_encoded'] == user_id]
            
            if user_history.empty:
                self.logger.warning("No purchase history found for user")
                return None
            
            # Calculate similarities
            indices = user_history.index.tolist()
            cosine_sim = cosine_similarity(tfidf_matrix[indices], tfidf_matrix)
            
            # Get recommendations
            similarity_scores = list(enumerate(cosine_sim[-1]))
            similarity_scores = [(i, score) for (i, score) in similarity_scores 
                               if i not in indices]
            
            # Sort and get top recommendations
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], 
                                    reverse=True)[:n_recommendations]
            
            # Create recommendations dataframe
            recommendations = pd.DataFrame({
                'product_name': [self.df.iloc[i[0]]['product_name'] for i in similarity_scores],
                'similarity_score': [i[1] for i in similarity_scores],
                'price': [self.df.iloc[i[0]]['discounted_price'] for i in similarity_scores],
                'rating': [self.df.iloc[i[0]]['rating'] for i in similarity_scores]
            })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return None

    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report."""
        self.logger.info("Generating insights report...")
        
        insights = []
        
        # Category insights
        top_categories = self.df['main_category'].value_counts().head(3)
        insights.append(f"Top 3 categories: {', '.join(top_categories.index)}")
        
        # Rating insights
        avg_rating = self.df['rating'].mean()
        insights.append(f"Average product rating: {avg_rating:.2f}")
        
        # Price insights
        avg_discount = self.df['discount_percentage'].mean() * 100
        insights.append(f"Average discount: {avg_discount:.1f}%")
        
        # Review insights
        avg_reviews = self.df['rating_count'].mean()
        insights.append(f"Average number of reviews: {avg_reviews:.0f}")
        
        return "\n".join(insights)

# Example usage:
if __name__ == "__main__":
    analyzer = AmazonProductAnalyzer("amazon.csv")
    
    # Get category analysis
    category_analysis = analyzer.analyze_categories()
    
    # Get rating analysis
    rating_analysis = analyzer.analyze_ratings()
    
    # Get pricing analysis
    pricing_analysis = analyzer.analyze_pricing()
    
    # Generate recommendations for a user
    recommendations = analyzer.generate_recommendations(user_id=893)
    
    # Generate insights report
    insights = analyzer.generate_insights_report()
    print(insights)
