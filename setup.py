from setuptools import setup, find_packages

setup(
    name="finance-rag-assistant",
    version="1.0.0",
    description="AI-powered payment reconciliation system",
    author="Keerthi Malathkar",
    author_email="keerthimalathkar@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.13",
        "chromadb>=0.4.22",
        "sentence-transformers>=2.3.1",
        "pandas>=2.1.4",
        "numpy>=1.26.3",
        "openpyxl>=3.1.2",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.12.8",
    ],
)