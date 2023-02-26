# e-assistant
Have you ever wished you could save time responding to emails by having an AI system write the response for you? Well, now you can! With this project, you can build an automatic email response generator that can generate responses to new emails in just a few seconds.

## Description
This project uses a sequence-to-sequence model with an LSTM encoder and decoder to generate automatic email responses. The model is trained on a dataset of email conversations and uses a combination of the input email and the previous emails in the conversation to generate the response. The model is implemented in PyTorch and the dataset is processed using Spacy.

## Dataset
The dataset used in this project is a collection of email conversations that have been anonymized and preprocessed. The dataset consists of pairs of emails, where the input is the current email and the output is the response to that email. The dataset is split into training, validation, and testing sets.

## Model Architecture
The model architecture consists of an LSTM encoder and decoder with attention. The encoder takes in the input email and produces a context vector that represents the input email. The decoder uses the context vector and the previous emails in the conversation to generate the response. The model is trained using teacher forcing, where the decoder is given the correct response at each time step. The model is trained using the Adam optimizer and the cross-entropy loss function. 

## Requirements
Python 3.6 or higher
PyTorch
Spacy

## Usage
1. Clone the repository: `https://github.com/Mahmood-Hussain/e-assistant.git`
2. Install the requirements (releasing soon)
3. Download and preprocess the dataset (releasing soon)
4. Train the model: python train.py
5. Generate responses: python train.py --inference_mode True



