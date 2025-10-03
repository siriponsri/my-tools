from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kaggle-utils",
    version="1.0.0",
    author="Kaggle Beginner Tools Contributors",
    author_email="",
    description="ชุดเครื่องมือสำหรับมือใหม่หัดแข่ง Kaggle - Universal toolkit for Kaggle competitions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/siriponsri/my-tools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="kaggle machine-learning data-science competition",
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "tqdm>=4.60.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "full": [
            "lightgbm>=3.0.0",
            "xgboost>=1.3.0",
            "catboost>=0.24.0",
            "optuna>=2.10.0",
        ],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)