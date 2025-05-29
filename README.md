# Digital Twin AI Assistant – Personalized GPT Agent with AWS Retraining Pipeline

## Purpose
**Inspiration**:
While enrolled in my first semester as a Master’s student in Artificial Intelligence, I wanted a way to complement my machine learning course. This project allowed me to gain practical experience in Pytorch. 

The goal was to create a self-reflective AI assistant built from scratch. It learns how I think, adapts to how I speak, and improves over time with an automated AWS fine-tuning loop. Eventually, it will become a digital twin that uses a journal of my life to interact as a personalized assistant.

## Description
The project provides a complete pipeline for:


**Custom GPT Training** – Transformer-based language model created in PyTorch and trained on personalized data.


**Chatbot Inference** – Lightweight text generation interface for real-time, local interaction.


**Vocabulary Encoding** – Tokenizer-style utility using a custom vocabulary.


**AWS SageMaker Integration** – Cloud-based retraining pipeline for continual fine-tuning.



## Architecture & Design
The flow is described below:


Model Training (training.py)
Trains a GPT-style model using tokenized text and positional embeddings. Generates model weights (model-01.pth).


Chatbot Interface (chatbot.py)
Loads the trained model and provides an interactive terminal-based chatbot using real-time sampling and decoding.


AWS Retraining Pipeline (sagemaker.py)
Launches a SageMaker pipeline that handles model training, registration, and output versioning. Designed for continuous self-fine-tuning.



## Data Flow


**Input**:
Custom journal-style text (starting with Wizard of Oz sample text), preprocessed into token sequences.


**Processing**:
Encode and tokenize vocabulary
Train transformer model
Deploy pipeline for cloud-based updates

**Output**:
Local chatbot generation
Saved model weights
SageMaker-registered models

## Getting Started

# Install Dependencies
```bash
pip install torch boto3 sagemaker
``` 

# Clone the Repository
```bash
git clone https://github.com/AaronRodriguez1/digital-twin-assistant.git
cd digital-twin-assistant
``` 

## Usage
1. Train the GPT Model
Train locally on a small dataset like Wizard of Oz:
```bash
python training.py -batch_size 32
``` 
2. Launch the Chatbot Interface
Interact with the trained model in real time:
```bash
python chatbot.py 
``` 

##Example Interaction:
```
Prompt: Hello, who are you?  
Response: I am your digital twin, learning from you each day...
```

## Authors

Aaron Rodriguez

## Version History

* 0.1
    * Initial Release with Wizard of OZ sample text
