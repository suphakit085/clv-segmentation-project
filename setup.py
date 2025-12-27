"""
Setup script for CLV Segmentation package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if exists
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Core dependencies
requirements = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "streamlit>=1.25.0",
    "scipy>=1.11.0",
]

setup(
    name="clv-segmentation",
    version="1.0.0",
    author="Suphakit",
    author_email="suphakit@example.com",
    description="Customer Lifetime Value prediction and segmentation toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suphakit085/clv-segmentation-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "clv-dashboard=dashboard.app:main",
        ],
    },
)

