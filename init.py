"""Main package initialization"""
from src.recommender.deep_learning import DeepRecommender
from src.recommender.behavior_tracker import RealTimeBehaviorTracker
from src.recommender.price_optimizer import PriceOptimizer
from src.recommender.video_analyzer import VideoAnalyzer
from src.recommender.notification_system import PersonalizedNotificationSystem

__version__ = "1.0.0"
__author__ = "Mini"

__all__ = [
    "DeepRecommender",
    "RealTimeBehaviorTracker",
    "PriceOptimizer",
    "VideoAnalyzer",
    "PersonalizedNotificationSystem",
]
