from setuptools import setup, find_packages

setup(
    name="FinSentinel",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
        "openai",
        "beautifulsoup4",
        "requests",
        "rich",
        "schedule",
        "yfinance",
        "tqdm",
    ],
    description="An end-to-end Chinese financial NLP pipeline: from LLM-assisted dataset construction to FinBERT sentiment analysis.",
    author="Your Name",
    python_requires=">=3.8",
)
