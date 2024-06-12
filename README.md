
# MD Chatbot

## Overview

Welcome to the MD Chatbot project! This repository contains a comprehensive setup for fine-tuning a LLaMA model to provide medical advice through a conversational AI. The project is structured to facilitate both local development and deployment via a Flask API.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [Local Model Execution](#local-model-execution)
- [Project Structure](#project-structure)
- [Model & Training](#model--training)
- [Data Processing](#data-processing)
- [Optimization](#optimization)
- [Deployment](#deployment)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JamieDeveloper/md-chatbot-flask
   cd md-chatbot-flask
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the API

1. **Set up your environment**: Ensure you have an API key set as an environment variable:

   ```bash
   export API_KEY=your-fixed-api-key
   ```

2. **Start the Flask application**:

   ```bash
   gunicorn app:app
   ```

3. **Send a request to the API**: Use a tool like `curl` or Postman to send a POST request to the `/process` endpoint with your question.

   ```bash
   curl -X POST http://localhost:5000/process -H "x-api-key: your-fixed-api-key" -H "Content-Type: application/json" -d '{"data": "What is the treatment for diabetes?"}'
   ```

### Local Model Execution

If you prefer to run the model locally or within a Jupyter notebook, use the `pure_model.py` script:

1. **Run the script**:

   ```bash
   python pure_model.py
   ```

2. **Interact with the model**: Input your question when prompted and receive the model's generated response.

## Project Structure

- `app.py`: Flask API setup for deploying the chatbot.
- `pure_model.py`: Script for local model execution and interaction.
- `requirements.txt`: List of dependencies required for the project.

## Model & Training

- Utilized Transformers and trl for fine-tuning a LLaMA model.
- Enhanced with 4-bit quantization via BitsAndBytes for efficiency.

## Data Processing

- Integrated Datasets library to load and preprocess a specialized medical dataset from Hugging Face.

## Optimization

- Implemented LoRA with peft for efficient parameterization.
- Tuning Custom training loop to fine-tune the model effectively.

## Deployment

- Used A text generation pipeline for responsive medical advice generation through a Flask API.
