# Federated Learning for Privacy-Preserving Medical Image Analysis

## 🧠 Overview
This project implements a Federated Learning framework to classify retinal (fundus) images for eye disease diagnosis while ensuring patient data privacy. Hospitals (clients) train models locally and only share model updates, not sensitive image data.

## 🚀 Key Features
- CNN-based classification of retinal diseases (e.g., glaucoma, diabetic retinopathy)
- Advanced preprocessing: circular cropping, Gaussian blurring, and CLAHE
- Federated Matched Averaging (FedMA) for neuron-level model aggregation
- Privacy-preserving multi-client simulation using local datasets
- Close-to-centralized accuracy without sharing medical data

## 🏗️ Architecture
- **Clients**: Locally train CNNs on private datasets
- **Server**: Aggregates client weights using FedMA
- **Model**: Keras CNN with 3 convolutional blocks and softmax output
- **Communication**: Periodic model update exchanges without raw data

## 📊 Results
- Accuracy: 87.5% (FedMA), vs. 85.2% (FedAvg), 89% (Centralized)
- Communication cost reduced by 18%
- Maintained high performance across non-IID clients

## ⚙️ Setup
```bash
pip install tensorflow opencv-python numpy
