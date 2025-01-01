# End-to-End Deep Learning Pipeline for Facial Emotion Classification with CI/CD Deployment on AWS
![Facial Emotion Types](emotionImage.png)

This repository contains an end-to-end deep learning web application that classifies facial emotion images as either Angry, Contempt, Disgust, Fear, Happy, Neutral, Surprised or Sad. The application is built with Python and Flask, using a pre-trained VGG16 model fine-tuned on a dataset of facial emotion images. It is designed for easy deployment on AWS infrastructure with continuous integration and deployment (CI/CD) using GitHub Actions.

---
## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Directory Structure](#directory-structure)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Pipeline Stages](#pipeline-stages)
- [Setup and Installation](#setup-and-installation)
- [AWS Deployment](#deployment)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview
Facial emotion recognition deep learning web applications have diverse use cases and applications across various industries. In healthcare, they can be used for mental health monitoring, helping therapists understand patients' emotional states during consultations. In customer service, businesses can deploy these apps to analyse customer emotions during interactions, enabling real-time adjustments to improve customer satisfaction. In education, such applications can monitor student engagement in virtual learning environments, identifying when learners are confused or disengaged. Entertainment platforms can use emotion recognition to personalise content recommendations based on user reactions. Additionally, in security and surveillance, these applications can detect suspicious behaviors or stress indicators, aiding in threat detection. Other applications include market research, where emotion recognition helps gauge consumer reactions to advertisements, and human-computer interaction, enhancing the responsiveness of virtual assistants and gaming experiences by adapting to users' emotions.

The web application allows users to upload an image of a facial emotion, which is then classified as one of eight types:
- Angry
- Contempt
- Disgust
- Fear
- Happy
- Neutral
- Surprised
- Sad

This project demonstrates:
1. Building and training a facial emotion recognition model using **transfer learning** with the VGG16 model pre-trained on the ImageNet dataset.
2. Implementing a scalable and reproducible **machine learning pipeline** with tools like DVC, MLFlow, and Dagshub for data versioning, experiment tracking, and model management.
3. Developing a web application using Flask to classify facial emotions from user-uploaded images.
4. Deploying the application to AWS using a Dockerised CI/CD pipeline powered by GitHub Actions, AWS ECR, and AWS EC2.


## Dataset
The dataset used for training and evaluating the model was sourced from [Kaggle](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition). It contains images representing eight distinct facial emotions: **anger**, **contempt**, **disgust**, **fear**, **happiness**, **neutral**, **sadness**, and **surprise**. The dataset features a diverse set of individuals, spanning various genders, ethnicities, and age groups, ensuring a comprehensive representation of human emotions. This diversity enhances the model's applicability across a wide range of real-world scenarios. 

For efficient storage and retrieval, the dataset was compressed and uploaded to an **AWS S3 bucket**. During the pipeline, the data was downloaded from S3 for model training and evaluation. The dataset is organized into `train` and `test` folders, each containing eight subdirectories named after the respective emotions. Each subdirectory includes images corresponding to that specific emotion, facilitating structured and efficient training and evaluation processes.

## Directory Structure

```
.
├── .github/                  
│   └── workflows/
│       └── main.py
├── artifacts/
│   ├── data_ingestion/
│   ├── prepare_base_model/
│   └── training/
├── config/
│   └── config.yaml
├── log/
│   └── running_logs.log        
├── research/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_trainer.ipynb
│   └── 04_model_evaluation_with_mlflow.ipynb
├── src/                  
│   └── emotionRecognition/
│       ├── components/
│       ├── config/
│       ├── constants/
│       ├── entity/
│       ├── pipeline/
│       └── utils
├── templates/                        # HTML templates               
│   └── index.html
├── app.py                            # Main Flask application
├── Dockefile                         # Docker configuration file    
├── dvc.yaml                          # DVC pipeline stages
├── main.py            
├── params.yaml              
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── scores.json      
├── setup.py             
└── template.py             
```

## Features

- **Deep Learning Model**: Transfer learning with VGG16 to classify facial emotions.
- **Reproducible ML Pipeline**: DVC for data ingestion, model training, and evaluation.
- **Experiment Tracking**: MLFlow and Dagshub for managing deep learning experiments.
- **Web Application**: Flask-based interface for emotion recognition.
- **CI/CD Pipeline**: Automated deployment using GitHub Actions and AWS infrastructure.
- **Scalable Cloud Deployment**: Hosted on AWS EC2 with Dockerised deployment.

## Technologies Used

- **Programming Language**: Python
- **Deep Learning Frameworks**: TensorFlow, Keras
- **Pipeline Tools**: DVC, MLFlow, Dagshub
- **Cloud Services**: AWS IAM, AWS S3, AWS ECR, AWS EC2
- **Web Framework**: Flask
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Version Control**: Git

## Pipeline Stages

### DVC Stages:

1. **Data Ingestion**: Fetch data from an S3 bucket.
2. **Base Model Development**: Modify VGG16 for emotion classification.
3. **Model Training**: Train the updated model on the ingested dataset.
4. **Model Evaluation with MlFlow Integration**: Evaluate the trained model on the validation dataset and experiment tracking.

### Experiment Tracking:

- MLFlow and Dagshub were used for:
    - Tracking hyperparameters, metrics, and artifacts.
    - Visualising training progress and evaluation results.

## Setup and Installation

### Prerequisites
- Python 3.10+
- Docker
- AWS account with access to IAM, S3, ECR, and EC2
- GitHub Account

### Local Setup
1. **Clone the repository**:
```
git clone https://github.com/OlumideOlumayegun/Facial-Emotion-Recognition-Using-MLflow-and-DVC.git
cd Facial-Emotion-Recognition-Using-MLflow-and-DVC
```
2. **Create a conda virtual environment**
```
conda create -n venv python=3.10 -y
conda activate venv
```
2. **Install dependencies**:
```
pip install -r requirements.txt
```
3. **Run the application locally**:
```
python app.py
```
Visit http://localhost:8080 in your browser to access the application.

### Docker Setup
1. **Build the Docker image**:
```
docker build -t emotion .
```
2. **Run the Docker container**:
```
docker run -p 8080:8080 emotion
```
## AWS Deployment

### Infrastructure Setup

The application is hosted on AWS infrastructure, with the following setup:
1. **IAM User**: Create an IAM user with specific policies for EC2 (AmazonEC2FullAccess) and ECR (AmazonEC2ContainerRegistryFullAccess).
2. **ECR Repository**: Create an ECR repository to store the Docker image..
3. **EC2 Instance**: Launch an EC2 instance with Docker installed and configure it as a self-hosted GitHub Actions runner.

### CI/CD with Github Actions

**GitHub Secrets** are used to securely store AWS credentials, and changes pushed to the main branch automatically trigger the deployment pipeline.
Add the following GitHub secrets for AWS credentials:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION
- ECR_REPOSITORY_NAME
- AWS_ECR_LOGIN_URI

A **GitHub Actions** workflow is configured to automate the CI/CD pipeline. On pushing changes to the repository, GitHub Actions triggers the workflow to:
- Build the Docker image.
- Push the Docker image to AWS ECR.
- Launch or update the EC2 instance.
- Pull the latest Docker image from ECR and run it on EC2. 

## Usage
Once the application is deployed, users can upload facial emotion images through the web interface. The application classifies each uploaded image as one of the eight emotion types and displays the result.

## Future Work
While the current implementation demonstrates promising results in facial emotion recognition, several areas can be explored to further enhance the model's performance and applicability:
- Increased dataset size to enhance accuracy across various use cases
- Enhance the classification model by testing other CNN architectures.
- Optimize model deployment with multi-instance load balancing.
- Improve UI/UX of the web application.
- Implement additional security measures for production-level deployment.

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


