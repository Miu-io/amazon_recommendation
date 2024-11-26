"""setup.py for project configuration"""
from setuptools import setup, find_packages

setup(
    name="amazon-advanced-recommender",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.8.0",
        "torch>=1.9.0",
        "transformers>=4.18.0",
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.5",
        "pillow>=9.0.0",
        "nltk>=3.7.0",
        "spacy>=3.2.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "plotly>=5.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
    },
    python_requires=">=3.8",
    author="Mini",
    author_email="DM me",  # Changed to match support contact method
    description="Advanced Amazon product recommendation system with deep learning capabilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
