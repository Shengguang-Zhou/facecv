"""FaceCV - 专业的人脸识别框架

用于人脸识别考勤、陌生人检测、多人脸识别等场景
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="facecv",
    version="0.1.0",
    author="EurekAI Lab",
    author_email="ai@eurekai.com",
    description="专业的人脸识别框架，支持InsightFace和DeepFace",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EurekAI-Lab/FaceCV",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "python-multipart>=0.0.5",
        "aiofiles>=0.7.0",
        "insightface>=0.7.0",
        "deepface>=0.0.75",
        "onnxruntime>=1.8.0",
        "scikit-learn>=0.24.0",
        "pymongo>=3.12.0",
        "sqlalchemy>=1.4.0",
        "chromadb>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "pre-commit>=2.13",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "facecv=facecv.cli:main",
        ],
    },
)