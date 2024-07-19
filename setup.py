from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fastdf",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A high-performance DataFrame implementation built on top of NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stwrn/fastdf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov", "black", "isort"],
    },
)
