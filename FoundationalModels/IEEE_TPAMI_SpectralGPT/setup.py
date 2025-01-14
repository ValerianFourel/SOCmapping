from setuptools import setup, find_packages

setup(
    name="IEEE_TPAMI_SpectralGPT",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Spectral-based GPT model implementation for IEEE TPAMI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/IEEE_TPAMI_SpectralGPT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "transformers>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
        ],
    },
)
