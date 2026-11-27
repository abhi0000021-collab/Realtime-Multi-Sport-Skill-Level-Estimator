# Real-Time Multi-Sport Skill Level Estimator

An AI-powered web application that analyzes an athlete's uploaded sports video using human pose estimation and machine learning. It estimates a performance score and skill level, provides improvement feedback, and generates an annotated video and PDF report.

## Overview

The application combines a React frontend with a Python/FastAPI backend. The backend processes uploaded video frames, extracts pose-based movement features, uses trained machine-learning models for performance estimation, and returns analysis results to the frontend.

## Features

- Upload athlete videos in **MP4, AVI, MOV, MKV, or WEBM** formats.
- Detect human body keypoints using a YOLOv8 pose model exported to ONNX.
- Extract movement features, including joint angles and movement speed.
- Estimate a performance score and skill level using a trained machine-learning model.
- Generate coaching feedback, improvement suggestions, and weakness insights.
- Produce an annotated MP4 video with pose-skeleton visualization.
- Generate a downloadable PDF performance report.
- Track player performance history and display leaderboard information.

## Technology Stack

| Area | Technologies |
| --- | --- |
| Frontend | React, JavaScript, CSS |
| Backend / API | Python, FastAPI, Uvicorn |
| Pose estimation | YOLOv8 Pose (ONNX), ONNX Runtime |
| Computer vision | OpenCV, NumPy |
| Machine learning | Scikit-learn, Joblib |
| Data handling | Pandas |
| PDF reports | ReportLab |
| Video conversion | FFmpeg |
| Deployment | Render |

## Project Structure

```text
Realtime-Multi-Sport-Skill-Level-Estimator/
├── backend/
│   └── main.py
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
├── datasets/
│   └── skill_data.csv
├── models/
│   ├── skill_model.pkl
│   └── yolov8s-pose.onnx
├── videos/
├── train_model.py
├── requirements.txt
├── .python-version
└── README.md
```

## Getting Started

### Prerequisites

Install the following before running the application:

- Python **3.11**
- Node.js and npm
- FFmpeg, available on your system PATH

### 1. Clone the repository

```bash
git clone https://github.com/abhi0000021-collab/Realtime-Multi-Sport-Skill-Level-Estimator.git
cd Realtime-Multi-Sport-Skill-Level-Estimator
```

### 2. Set up and run the backend

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Make sure the two model files are present in the `models/` directory:

- `skill_model.pkl`
- `yolov8s-pose.onnx`

Start the API server from the repository root:

```bash
python -m uvicorn backend.main:app --reload
```

The API documentation is available at:

[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)

### 3. Set up and run the frontend

Open a second terminal and move into the frontend directory:

```bash
cd frontend
npm install
```

For local development, the frontend uses `http://127.0.0.1:8000` as its default backend URL. To use a different backend URL, set the `REACT_APP_API_URL` environment variable before starting or building the React app.

PowerShell example:

```powershell
$env:REACT_APP_API_URL="http://127.0.0.1:8000"
npm start
```

Start the frontend:

```bash
npm start
```

The React development server normally runs at [`http://localhost:3000`](http://localhost:3000).

## How It Works

1. Enter the player's name and upload a supported sports video.
2. The FastAPI backend reads the video and processes its frames.
3. The ONNX pose model detects body keypoints in sampled frames.
4. The application extracts movement features and estimates a performance score.
5. The backend derives a skill level and prepares feedback and player-performance information.
6. The application creates an annotated video and a PDF report.
7. The React dashboard displays the analysis and provides links to the generated files.

## Training the Skill Prediction Model

The repository includes a training script and sample dataset. From the repository root, run:

```bash
python train_model.py
```

The script reads `datasets/skill_data.csv`, uses the `score` column as the target, trains a `RandomForestRegressor`, and saves the model to `models/skill_model.pkl`.

## API

The video analysis endpoint is:

- `POST /upload-video/`

It accepts a video file in the `file` form field and a player name in the `player_name` form field. See the interactive FastAPI documentation at `/docs` for the request and response schema.

## Notes

- FFmpeg must be installed separately; it is not installed by the Python requirements file.
- The backend currently writes uploaded videos and generated reports to local `videos/` and `outputs/` directories. On hosting platforms with ephemeral filesystems, generated files may not remain available after an instance restart. Configure persistent or object storage before relying on long-term file availability.
- The included `datasets/skill_data.csv` is a small sample dataset. Model predictions depend on the data used for training and should not be treated as a certified sports assessment.

## Author

**Abhijith**

B.Sc. Artificial Intelligence and Machine Learning

GitHub: [abhi0000021-collab](https://github.com/abhi0000021-collab)

---

If you find this project useful, feel free to explore the source code and follow its development.
