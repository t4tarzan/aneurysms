from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="aneurysm_detection",
    version="0.1.0",
    description="Anatomy-based post-processing for intracranial aneurysm detection",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "black>=21.12b0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
