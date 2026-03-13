---
title: AI Video Classifier
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# AI Video Classifier

This project is an AI Video Classifier that allows users to upload Instagram videos and classifies them as either AI-generated or real using the XceptionNet model. The application also incorporates user feedback to improve the model's accuracy over time.

## Project Structure

```
ai-video-classifier
├── app.py                   # Main entry point of the application
├── models
│   ├── xception_model.py    # Implementation of the XceptionNet model
│   └── feedback_trainer.py   # User feedback management and model retraining
├── static
│   └── uploads              # Directory for temporarily storing uploaded videos
├── templates
│   ├── index.html           # User interface for video upload and results display
│   └── result.html          # Template for displaying classification results
├── requirements.txt         # Python dependencies for the project
└── README.md                # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd ai-video-classifier
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`. Then, install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Start the Flask application by running:
   ```
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5000`.

## Usage Guidelines

- Navigate to the application in your web browser.
- Use the provided interface to upload an Instagram video.
- After processing, the application will display whether the video is classified as AI-generated or real.
- Users can provide feedback on the classification, which will be used to retrain the model.

## Model Information

The XceptionNet model is utilized for video classification in this project. It has been trained to distinguish between AI-generated and real videos. The model's performance can be improved through user feedback, which is collected and processed to enhance its accuracy.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.