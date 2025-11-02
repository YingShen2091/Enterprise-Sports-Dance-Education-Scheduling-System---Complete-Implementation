"""
Setup configuration for Sports Dance Education Scheduling System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="sports-dance-scheduling",
    version="3.0.0",
    author="Enterprise Development Team",
    author_email="dev@sportsdancescheduling.com",
    description="Enterprise Sports Dance Education Scheduling System with AI optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sports-dance-scheduling",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/sports-dance-scheduling/issues",
        "Documentation": "https://sports-dance-scheduling.readthedocs.io",
        "Source Code": "https://github.com/yourusername/sports-dance-scheduling",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.1",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.1.1",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.2",
            "seaborn>=0.12.2",
            "plotly>=5.15.0",
        ],
        "distributed": [
            "ray[tune]>=2.6.1",
            "horovod>=0.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sports-dance-scheduler=sports_dance_scheduling:main",
            "sds-train=sports_dance_scheduling.train:main",
            "sds-api=sports_dance_scheduling.api:main",
            "sds-evaluate=sports_dance_scheduling.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "sports_dance_scheduling": [
            "config/*.yaml",
            "config/*.json",
            "data/*.csv",
            "models/*.onnx",
        ],
    },
    zip_safe=False,
    keywords=[
        "scheduling",
        "optimization",
        "machine-learning",
        "deep-learning",
        "reinforcement-learning",
        "education",
        "sports",
        "dance",
        "multi-objective-optimization",
        "neural-networks",
    ],
)