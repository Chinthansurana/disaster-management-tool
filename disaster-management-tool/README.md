# Disaster Management Tool

This project builds an automated disaster management tool using Siamese Neural Networks and Encoder-Decoder Architectures for building damage assessment from satellite images. The system achieves 85% classification accuracy and helps speed up disaster recovery planning.

## Features

- **Siamese Neural Networks** for comparing satellite images and assessing building damage.
- **Encoder-Decoder Architecture** for pixel-level damage detection.
- **85% Classification Accuracy** for damage assessment.
- **Scalable and containerized** using Docker.
- **API deployment** with FastAPI for easy integration.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Chinthansurana/disaster-management-tool.git
   cd disaster-management-tool
2. Install dependencies:
    pip install -r requirements.txt
3. Set up Redis on your machine (for caching or message queue purposes):
    Install Redis from redis.io
    Start Redis server:
            redis-server