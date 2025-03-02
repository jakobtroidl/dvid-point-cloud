from setuptools import setup, find_packages

setup(
    name="dvid-point-cloud",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "protobuf>=3.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "requests-mock>=1.9.0",
            "pytest-cov>=2.11.0",
            "flake8>=3.8.0",
            "mypy>=0.790",
        ],
    },
    python_requires=">=3.7",
    author="",
    author_email="",
    description="Library for creating point clouds for sparse volumes within DVID",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)