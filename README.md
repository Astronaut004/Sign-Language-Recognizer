# Sign Language Recognizer

## Introduction

Sign Language Recognizer is a real-time hand sign recognition system that utilizes computer vision and machine learning to interpret and predict hand signs. This project provides a user-friendly interface for users to interact with, making sign language recognition accessible and efficient.

## Features

- Real-time hand sign recognition using a webcam
- Supports a variety of hand signs
- Provides instant predictions for recognized signs
- User-friendly interface

## Prerequisites

Before you begin, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- TensorFlow
- Flask
- Socket.IO

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/sign-language-recognizer.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd sign-language-recognizer
    ```

3. **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    ```

4. **Activate the virtual environment:**

    - On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

5. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Flask application to start the server:**

    ```bash
    python app.py
    ```

2. **Open a web browser and navigate to `http://localhost:5000` to access the Sign Language Recognizer.**

3. **Press the "Start Recognition" button to initiate the webcam and begin hand sign recognition.**

4. **Interact with the application, and it will provide real-time predictions for recognized hand signs.**

## Project Structure

- `fron.py`: Main Flask application file.
- `templates/`: Folder containing html and css file.
- `Model/`: Folder containing the trained machine learning model.


