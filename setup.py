"""Setup script for ui-vlm-training package."""

import os
from setuptools import setup, find_packages

setup(
    name="ui-vlm-training",
    version="0.1.0",
    description="Vision-Language Model training for UI interaction tasks",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/ui-vlm-training",
    packages=find_packages(include=["src*", "config*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "openai>=1.0.0",
        "anthropic>=0.8.0",
        "google-generativeai>=0.3.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "pandas>=2.0.0",
        "google-cloud-compute>=1.14.0",
        "google-cloud-storage>=2.10.0",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        "gymnasium>=0.28.0",
        "pyyaml>=6.0",
        "hydra-core>=1.3.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "matplotlib>=3.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ui-train=scripts.train:main",
            "ui-eval=scripts.eval:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
