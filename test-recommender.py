"""Test cases for the recommendation system"""
import pytest
import numpy as np
from src.recommender.deep_learning import DeepRecommender
from src.recommender.behavior_tracker import RealTimeBehaviorTracker
from src.recommender.price_optimizer import PriceOptimizer

class TestDeepRecommender:
    @pytest.fixture
    def recommender(self):
        return DeepRecommender(embedding_dim=64)
        
    def test_user_embedding(self, recommender):
        user_data = {
            "user_history": np.random.randn(10, 64),
            "user_demographics": np.random.randn(10),
            "behavior_sequence": np.random.randn(20, 50)
        }
        embedding = recommender._build_user_tower()(user_data)
        assert embedding.shape == (1, 64)
        
    def test_item_embedding(self, recommender):
        item_data = {
            "image_features": np.random.randn(2048),
            "text_features": np.random.randn(768),
            "categorical_features": np.random.randn(50)
        }
        embedding = recommender._build_item_tower()(item_data)
        assert embedding.shape == (1, 64)
        
    def test_recommendation_generation(self, recommender):
        user_id = "test_user"
        recommendations = recommender.generate_recommendations(user_id, n=5)
        assert len(recommendations) == 5
        assert all(isinstance(r, dict) for r in recommendations)

class TestBehaviorTracker:
    @pytest.fixture
    def tracker(self):
        return RealTimeBehaviorTracker()
        
    def test_event_tracking(self, tracker):
        user_id = "test_user"
        event_type = "view_product"
        metadata = {"product_id": "test_product"}
        
        tracker.track_event(user_id, event_type, metadata)
        patterns = tracker.behavior_patterns[user_id]
        
        assert "frequencies" in patterns
        assert event_type in patterns["frequencies"]
        assert patterns["frequencies"][event_type] == 1

class TestPriceOptimizer:
    @pytest.fixture
    def optimizer(self):
        return PriceOptimizer()
        
    def test_price_optimization(self, optimizer):
        product_id = "test_product"
        current_price = 100.0
        
        # Add some test data
        optimizer.price_history[product_id] = [90.0, 95.0, 100.0]
        optimizer.sales_history[product_id] = [100, 90, 80]
        
        optimal_price = optimizer.optimize_price(
            product_id=product_id,
            current_price=current_price
        )
        
        assert isinstance(optimal_price, float)
        assert optimal_price > 0
        assert abs(optimal_price - current_price) <= current_price * 0.2  # Max 20% change

def test_end_to_end():
    # Initialize components
    recommender = DeepRecommender()
    tracker = RealTimeBehaviorTracker()
    optimizer = PriceOptimizer()
    
    # Test workflow
    user_id = "test_user"
    product_id = "test_product"
    
    # Track behavior
    tracker.track_event(
        user_id=user_id,
        event_type="view_product",
        metadata={"product_id": product_id}
    )
    
    # Generate recommendations
    recommendations = recommender.generate_recommendations(user_id)
    
    # Optimize price
    optimal_price = optimizer.optimize_price(product_id)
    
    # Assertions
    assert len(recommendations) > 0
    assert optimal_price > 0
    assert len(tracker.session_data[user_id]) == 1

if __name__ == "__main__":
    pytest.main([__file__])
