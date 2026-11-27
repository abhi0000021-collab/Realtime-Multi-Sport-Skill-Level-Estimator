
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import random
import shutil
import os
import cv2
import numpy as np
import onnxruntime as ort
import joblib
import subprocess
import pandas as pd
import json
import logging
import uuid

from pathlib import Path
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multisport-estimator")


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Real-Time Multi-Sport Skill Level Estimator"
)


# =====================================================
# CORS CONFIGURATION
# =====================================================

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_FILE = PROJECT_DIR / "models" / "skill_model.pkl"
MODEL_PATH = PROJECT_DIR / "models" / "yolov8s-pose.onnx"

UPLOAD_FOLDER = PROJECT_DIR / "videos"
OUTPUT_FOLDER = PROJECT_DIR / "outputs"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_FOLDER)),
    name="outputs"
)


# =====================================================
# LOAD TRAINED MODELS
# =====================================================

logger.info("Loading skill prediction model...")

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Skill model not found: {MODEL_FILE}"
    )

skill_model = joblib.load(MODEL_FILE)

logger.info("Loading ONNX pose estimation model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Pose model not found: {MODEL_PATH}"
    )

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

# Use the model's actual input name.
# This preserves compatibility with your ONNX model.
input_name = session.get_inputs()[0].name

logger.info("Models loaded successfully.")


# =====================================================
# PLAYER HISTORY
# =====================================================

PLAYERS_FILE = BASE_DIR / "players.json"


def load_players():

    if not PLAYERS_FILE.exists():
        return {}

    try:
        with open(PLAYERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except (json.JSONDecodeError, OSError):
        logger.exception("Unable to load player history.")
        return {}


def save_players(data):

    with open(PLAYERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# =====================================================
# POSE DETECTION
# =====================================================

def run_pose(frame):

    img = cv2.resize(frame, (640, 640))

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(np.float32) / 255.0

    img = np.transpose(
        img,
        (2, 0, 1)
    )

    img = np.expand_dims(img, 0)

    return session.run(
        None,
        {input_name: img}
    )


# =====================================================
# KEYPOINT EXTRACTION
# =====================================================

def extract_keypoints(outputs, frame_shape):

    h, w = frame_shape[:2]

    preds = outputs[0]

    preds = np.squeeze(preds).T

    if len(preds) == 0:
        return []

    best = preds[np.argmax(preds[:, 4])]

    if best[4] < 0.3:
        return []

    kpts = best[5:56].reshape(17, 3)

    keypoints = []

    for x, y, conf in kpts:

        if conf > 0.5:

            # Scale coordinates from 640x640
            # to original video dimensions.

            px = int(x / 640 * w)
            py = int(y / 640 * h)

            keypoints.append((px, py))

        else:
            keypoints.append((0, 0))

    return keypoints


# =====================================================
# SKELETON CONNECTIONS
# =====================================================

SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]


def draw_skeleton(frame, kpts):

    for x, y in kpts:

        if x > 0 and y > 0:

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    for p1, p2 in SKELETON:

        if (
            p1 < len(kpts)
            and p2 < len(kpts)
            and kpts[p1] != (0, 0)
            and kpts[p2] != (0, 0)
        ):

            cv2.line(
                frame,
                kpts[p1],
                kpts[p2],
                (255, 0, 0),
                2
            )

    return frame


# =====================================================
# ANGLE CALCULATION
# =====================================================

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) *
        np.linalg.norm(bc) + 1e-6
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    return float(
        np.degrees(np.arccos(cosine))
    )


def extract_angles(k):

    try:

        return {
            "right_elbow": calculate_angle(
                k[6], k[8], k[10]
            ),

            "left_elbow": calculate_angle(
                k[5], k[7], k[9]
            ),

            "right_knee": calculate_angle(
                k[12], k[14], k[16]
            ),

            "left_knee": calculate_angle(
                k[11], k[13], k[15]
            )
        }

    except (IndexError, TypeError, ValueError):
        return None


# =====================================================
# MOVEMENT SPEED
# =====================================================

def compute_speed(k, previous_center=None):

    hips = [k[11], k[12]]

    if (0, 0) in hips:
        return 0, previous_center

    center = np.mean(hips, axis=0)

    if previous_center is None:
        return 0, center

    speed = np.linalg.norm(
        center - previous_center
    )

    return float(speed), center


# =====================================================
# SKILL LEVEL
# =====================================================

def get_level(score):

    if score < 50:
        return "Beginner"

    elif score < 75:
        return "Intermediate"

    else:
        return "Advanced"


# =====================================================
# AI COACH COMMENTARY
# =====================================================

def ai_coach_commentary(score, level, tips):

    elite_comments = [
        "Outstanding control and technique. You move like a trained athlete.",
        "Excellent performance. Your body coordination is impressive.",
        "High-level execution. Maintain this intensity."
    ]

    good_comments = [
        "Strong performance overall. Small refinements will make you elite.",
        "Good technique. Focus on consistency for pro-level results.",
        "Nice control. Improve speed and you’ll level up quickly."
    ]

    avg_comments = [
        "Decent effort. Work on posture and timing.",
        "You have potential, but need better consistency.",
        "Focus on cleaner movement and balance."
    ]

    beginner_comments = [
        "You are learning. Focus on fundamentals and repetition.",
        "Train regularly to build muscle memory.",
        "Start slow and master correct form first."
    ]

    if score > 80:
        base = random.choice(elite_comments)

    elif score > 60:
        base = random.choice(good_comments)

    elif score > 40:
        base = random.choice(avg_comments)

    else:
        base = random.choice(beginner_comments)

    if tips:
        tip_line = f" Key focus: {tips[0]}."
    else:
        tip_line = ""

    return base + tip_line


# =====================================================
# AI PERFORMANCE FEEDBACK
# =====================================================

def generate_ai_comment(score, angles, speed):

    comments = []

    if angles:

        if angles["right_elbow"] < 45:

            comments.append(
                "Your arm extension needs improvement "
                "for stronger strokes."
            )

        else:

            comments.append(
                "Great elbow control — your technique "
                "looks refined."
            )

    if speed < 5:

        comments.append(
            "Try increasing your movement speed "
            "to improve performance."
        )

    else:

        comments.append(
            "Nice pace and movement — that gives you an edge."
        )

    if score > 80:

        comments.append(
            "You’re performing at a competitive athlete level."
        )

    elif score > 60:

        comments.append(
            "Solid performance, but there’s room to level up."
        )

    else:

        comments.append(
            "You’re building the basics — keep training consistently."
        )

    return " ".join(comments)


# =====================================================
# ATHLETE COMPARISON
# =====================================================

def athlete_comparison(score):

    if score > 85:

        return (
            "Your movement quality is approaching "
            "professional athlete standards."
        )

    elif score > 70:

        return (
            "Your form resembles trained club-level players."
        )

    else:

        return (
            "You’re on the path — consistent practice will elevate you."
        )


# =====================================================
# MOTIVATION
# =====================================================

def motivation_line():

    lines = [
        "Champions train when nobody is watching.",
        "Small improvements daily create elite athletes.",
        "Focus on progress, not perfection.",
        "Your only competition is yesterday’s version of you.",
        "Consistency beats intensity."
    ]

    return random.choice(lines)


# =====================================================
# AI INSIGHTS
# =====================================================

def ai_insight(scores):

    if len(scores) < 3:

        return "Train more sessions to unlock AI insights."

    trend = scores[-1] - scores[-3]

    if trend > 5:

        return "Your progress trend is excellent."

    elif trend > 0:

        return "You are improving gradually."

    else:

        return "Performance plateau detected."


# =====================================================
# TRAINING PLAN
# =====================================================

def training_plan(score):

    if score < 50:

        return "Focus on basic form drills."

    elif score < 75:

        return "Add agility and reaction drills."

    else:

        return "Maintain elite-level training."


# =====================================================
# COACH COMMENT
# =====================================================

def get_coach_comment(score):

    if score > 80:

        return (
            "Elite performance. You show strong control "
            "and confidence."
        )

    elif score > 60:

        return "Good work. Focus on consistency and speed."

    elif score > 40:

        return "Average form. Improve technique and posture."

    else:

        return "Beginner level. Train regularly to build fundamentals."


# =====================================================
# PDF REPORT GENERATION
# =====================================================

def create_pdf_report(
    pdf_path,
    player_name,
    avg_score,
    level,
    grade,
    technique_score,
    balance_score,
    speed_score,
    comparison,
    tips,
    coach_comment,
    motivation
):

    c = canvas.Canvas(
        str(pdf_path),
        pagesize=A4
    )

    page_width, page_height = A4

    dark = HexColor("#111827")
    green = HexColor("#22c55e")
    grey = HexColor("#374151")

    # HEADER

    c.setFillColor(dark)

    c.rect(
        0,
        page_height - 70,
        page_width,
        70,
        fill=1
    )

    c.setFillColorRGB(1, 1, 1)

    c.setFont("Helvetica-Bold", 18)

    c.drawString(
        45,
        page_height - 43,
        "AI SPORTS PERFORMANCE REPORT"
    )

    # PLAYER DETAILS

    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica", 11)

    date = dt.now().strftime("%d %b %Y")

    c.drawString(
        50,
        page_height - 100,
        f"Player Name: {player_name}"
    )

    c.drawString(
        50,
        page_height - 120,
        f"Date: {date}"
    )

    # SCORE BOX

    c.setFillColor(green)

    c.rect(
        50,
        page_height - 190,
        page_width - 100,
        50,
        fill=1
    )

    c.setFillColorRGB(1, 1, 1)

    c.setFont("Helvetica-Bold", 18)

    c.drawString(
        190,
        page_height - 172,
        f"Skill Score: {avg_score:.2f}"
    )

    # SKILL LEVEL

    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica-Bold", 12)

    c.drawString(
        50,
        page_height - 220,
        f"Skill Level: {level}"
    )

    c.drawString(
        330,
        page_height - 220,
        f"Grade: {grade}"
    )

    # PERFORMANCE BREAKDOWN

    c.setFont("Helvetica-Bold", 13)

    c.drawString(
        50,
        page_height - 255,
        "Performance Breakdown"
    )

    c.setFont("Helvetica", 11)

    c.drawString(
        65,
        page_height - 278,
        f"Technique: {technique_score:.1f}"
    )

    c.drawString(
        65,
        page_height - 298,
        f"Balance: {balance_score:.1f}"
    )

    c.drawString(
        65,
        page_height - 318,
        f"Speed: {speed_score:.1f}"
    )

    # PERFORMANCE INSIGHTS

    c.setFont("Helvetica-Bold", 13)

    c.drawString(
        50,
        page_height - 355,
        "Performance Insights"
    )

    def draw_wrapped_text(text, x, y, max_chars=85):

        text_obj = c.beginText(x, y)

        text_obj.setFont("Helvetica", 9)

        words = text.split()

        line = ""

        for word in words:

            if len(line) + len(word) + 1 <= max_chars:

                line += word + " "

            else:

                text_obj.textLine(line)

                line = word + " "

        if line:
            text_obj.textLine(line)

        c.drawText(text_obj)

        return text_obj.getY()

    y = page_height - 378

    for tip in tips[:6]:

        if y < 300:
            break

        y = draw_wrapped_text(
            "• " + tip,
            65,
            y
        )

        y -= 8

    # ATHLETE COMPARISON

    comparison_y = max(y - 10, 260)

    c.setFont("Helvetica-Bold", 12)

    c.drawString(
        50,
        comparison_y,
        "Athlete Comparison"
    )

    draw_wrapped_text(
        comparison,
        60,
        comparison_y - 18
    )

    # COACH FEEDBACK

    coach_y = max(comparison_y - 65, 175)

    c.setFont("Helvetica-Bold", 12)

    c.drawString(
        50,
        coach_y,
        "Coach Feedback"
    )

    draw_wrapped_text(
        coach_comment,
        60,
        coach_y - 18
    )

    # MOTIVATION BOX

    c.setFillColor(HexColor("#e8fdf5"))

    c.rect(
        50,
        65,
        page_width - 100,
        42,
        fill=1
    )

    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica-Bold", 10)

    c.drawString(
        60,
        91,
        "Motivation"
    )

    c.setFont("Helvetica-Oblique", 9)

    c.drawString(
        60,
        76,
        motivation[:100]
    )

    # FOOTER

    c.setFillColor(grey)

    c.rect(
        0,
        0,
        page_width,
        30,
        fill=1
    )

    c.setFillColorRGB(1, 1, 1)

    c.setFont("Helvetica", 8)

    c.drawCentredString(
        page_width / 2,
        11,
        "Generated by Real-Time MultiSport Skill Estimator"
    )

    c.save()


# =====================================================
# VIDEO UPLOAD AND ANALYSIS API
# =====================================================

@app.post("/upload-video/")
def upload_video(
    file: UploadFile = File(...),
    player_name: str = Form(...)
):

    request_id = uuid.uuid4().hex[:8]

    logger.info(
        "[%s] UPLOAD REQUEST RECEIVED",
        request_id
    )

    original_filename = os.path.basename(
        file.filename or ""
    )

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    allowed_extensions = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    ]

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Please upload MP4, AVI, MOV, MKV, or WEBM."
            )
        )

    player_name = player_name.strip()

    if not player_name:

        raise HTTPException(
            status_code=400,
            detail="Player name is required."
        )

    unique_id = uuid.uuid4().hex[:12]

    safe_filename = Path(original_filename).stem[:60]

    name = f"{safe_filename}_{unique_id}"

    file_path = UPLOAD_FOLDER / f"{name}{extension}"

    out_path = OUTPUT_FOLDER / f"annotated_{name}.mp4"

    converted_path = OUTPUT_FOLDER / f"converted_{name}.mp4"

    pdf_path = OUTPUT_FOLDER / f"report_{name}.pdf"

    cap = None
    out = None

    try:

        # -------------------------------------------------
        # SAVE UPLOADED VIDEO
        # -------------------------------------------------

        with open(file_path, "wb") as buf:

            shutil.copyfileobj(
                file.file,
                buf
            )

        logger.info(
            "[%s] VIDEO FILE SAVED: %s bytes",
            request_id,
            file_path.stat().st_size
        )

        if file_path.stat().st_size == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty."
            )

        # -------------------------------------------------
        # OPEN VIDEO
        # -------------------------------------------------

        cap = cv2.VideoCapture(
            str(file_path)
        )

        if not cap.isOpened():

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unable to open the uploaded video. "
                    "Please try another video."
                )
            )

        w = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        h = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps_value = cap.get(
            cv2.CAP_PROP_FPS
        )

        fps = fps_value if fps_value > 0 else 25.0

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        logger.info(
            "[%s] VIDEO OPENED: %sx%s, FPS=%s, FRAMES=%s",
            request_id,
            w,
            h,
            fps,
            frame_count
        )

        if w <= 0 or h <= 0:

            raise HTTPException(
                status_code=400,
                detail="Invalid video dimensions."
            )

        # -------------------------------------------------
        # CREATE OUTPUT VIDEO
        # -------------------------------------------------

        out = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        if not out.isOpened():

            raise HTTPException(
                status_code=500,
                detail="Unable to create annotated video."
            )

        # -------------------------------------------------
        # INITIALIZE ANALYTICS
        # -------------------------------------------------

        all_scores = []

        all_tips = []

        tech_list = []

        bal_list = []

        speed_list = []

        previous_center = None

        angles = None

        speed = 0.0

        processed_frames = 0

        # -------------------------------------------------
        # VIDEO FRAME PROCESSING
        # -------------------------------------------------

        logger.info(
            "[%s] STARTING VIDEO FRAME PROCESSING",
            request_id
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            processed_frames += 1

            # Process only 1 out of every 3 frames.
            FRAME_SKIP = 3

            if (processed_frames - 1) % FRAME_SKIP != 0:
                out.write(frame)
                continue

            logger.info(
                "[%s] PROCESSING FRAME %s / %s",
                request_id,
                processed_frames,
                frame_count
            )

            # ONNX POSE DETECTION

            outputs = run_pose(frame)

            kpts = extract_keypoints(
                outputs,
                frame.shape
            )

            if len(kpts) == 0:

                out.write(frame)

                continue

            angles = extract_angles(kpts)

            if angles:

                # MOVEMENT SPEED

                speed, previous_center = compute_speed(
                    kpts,
                    previous_center
                )

                # -------------------------------------------------
                # MODEL INPUT - PRESERVED
                # -------------------------------------------------

                feature_vector = [

                    angles["right_elbow"],

                    angles["left_elbow"],

                    angles["right_knee"],

                    angles["left_knee"],

                    speed

                ]

                df = pd.DataFrame(
                    [feature_vector],
                    columns=[
                        "right_elbow",
                        "left_elbow",
                        "right_knee",
                        "left_knee",
                        "speed"
                    ]
                )

                # SKILL MODEL PREDICTION

                prediction_score = skill_model.predict(df)[0]

                all_scores.append(
                    float(prediction_score)
                )

                # -------------------------------------------------
                # TECHNIQUE SCORE
                # -------------------------------------------------

                technique = np.mean([

                    angles["right_elbow"],

                    angles["left_elbow"]

                ])

                technique_score = max(
                    0,
                    min(100, technique)
                )

                # -------------------------------------------------
                # BALANCE SCORE
                # -------------------------------------------------

                balance = np.mean([

                    angles["right_knee"],

                    angles["left_knee"]

                ])

                balance_score = max(
                    0,
                    min(100, balance)
                )

                # -------------------------------------------------
                # SPEED SCORE
                # -------------------------------------------------

                speed_score_frame = min(
                    speed * 15,
                    100
                )

                tech_list.append(
                    technique_score
                )

                bal_list.append(
                    balance_score
                )

                speed_list.append(
                    speed_score_frame
                )

                # -------------------------------------------------
                # FINAL FRAME SCORE
                # -------------------------------------------------

                final_score = (

                    technique_score * 0.4

                    + balance_score * 0.35

                    + speed_score_frame * 0.25

                )

                final_score = (
                    final_score + prediction_score
                ) / 2

                # -------------------------------------------------
                # SMART FEEDBACK
                # -------------------------------------------------

                # RIGHT ELBOW

                if angles["right_elbow"] < 45:

                    all_tips.append(
                        "Your right elbow angle is too closed. "
                        "This reduces power generation. "
                        "Try extending your arm fully during "
                        "execution for stronger shots."
                    )

                # LEFT ELBOW

                if angles["left_elbow"] < 45:

                    all_tips.append(
                        "Left elbow positioning is limiting your "
                        "swing range. Keep your arm more open to "
                        "improve control and balance."
                    )

                # RIGHT KNEE

                if angles["right_knee"] < 140:

                    all_tips.append(
                        "Your right knee is not bending enough. "
                        "Proper knee bend helps in shock absorption "
                        "and stability."
                    )

                # LEFT KNEE

                if angles["left_knee"] < 140:

                    all_tips.append(
                        "Left knee mechanics need improvement. "
                        "A deeper bend improves agility and "
                        "quick direction changes."
                    )

                # SPEED

                if speed < 3:

                    all_tips.append(
                        "Your movement speed is low. Faster footwork "
                        "increases reaction time and competitive "
                        "performance."
                    )

                # POSTURE
                # FIXED: Use the score variables calculated above.

                if (
                    technique_score < 60
                    and balance_score < 60
                ):

                    all_tips.append(
                        "Your overall posture needs work. "
                        "Focus on body alignment and core stability "
                        "during play."
                    )

            # DRAW SKELETON

            frame = draw_skeleton(
                frame,
                kpts
            )

            out.write(frame)

        # -------------------------------------------------
        # RELEASE VIDEO RESOURCES
        # -------------------------------------------------

        cap.release()
        cap = None

        out.release()
        out = None

        logger.info(
            "[%s] FRAME PROCESSING COMPLETED. "
            "TOTAL FRAMES=%s, VALID POSES=%s",
            request_id,
            processed_frames,
            len(all_scores)
        )

        # -------------------------------------------------
        # CHECK POSE DETECTION
        # -------------------------------------------------

        if not all_scores:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No person or valid body pose detected in the video. "
                    "Please upload a clear video showing a person."
                )
            )

        # -------------------------------------------------
        # AVERAGE PERFORMANCE METRICS
        # -------------------------------------------------

        technique_score = float(
            np.mean(tech_list)
        )

        balance_score = float(
            np.mean(bal_list)
        )

        speed_score = float(
            np.mean(speed_list)
        )

        avg_score = float(
            np.mean(all_scores)
        )

        # -------------------------------------------------
        # FFMPEG VIDEO CONVERSION
        # -------------------------------------------------

        logger.info(
            "[%s] STARTING FFMPEG CONVERSION",
            request_id
        )

        ffmpeg_result = subprocess.run(

            [
                "ffmpeg",
                "-y",
                "-i", str(out_path),
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                str(converted_path)
            ],

            capture_output=True,

            text=True,

            timeout=180

        )

        if ffmpeg_result.returncode != 0:

            logger.error(
                "[%s] FFMPEG ERROR: %s",
                request_id,
                ffmpeg_result.stderr[-3000:]
            )

            raise HTTPException(
                status_code=500,
                detail="FFmpeg video conversion failed."
            )

        logger.info(
            "[%s] FFMPEG CONVERSION COMPLETED",
            request_id
        )

        # -------------------------------------------------
        # PLAYER HISTORY
        # -------------------------------------------------

        players = load_players()

        if player_name not in players:

            players[player_name] = []

        players[player_name].append({

            "date": dt.now().strftime("%d %b %Y"),

            "score": avg_score

        })

        save_players(players)

        history = players[player_name]

        # -------------------------------------------------
        # LEADERBOARD
        # -------------------------------------------------

        leaderboard = sorted(

            [
                {
                    "name": k,
                    "score": v[-1]["score"]
                }

                for k, v in players.items()

                if v
            ],

            key=lambda x: x["score"],

            reverse=True

        )[:5]

        # -------------------------------------------------
        # IMPROVEMENT TRACKING
        # -------------------------------------------------

        improvement_msg = ""

        if len(history) >= 2:

            last = history[-1]["score"]

            prev = history[-2]["score"]

            diff = last - prev

            percent = (
                (diff / prev) * 100
                if prev > 0
                else 0
            )

            if diff > 0:

                improvement_msg = (
                    f"{player_name} improved {percent:.1f}% "
                    "since last session 🔥"
                )

            elif diff < 0:

                improvement_msg = (
                    f"{player_name} dropped {abs(percent):.1f}% "
                    "— time to train harder 💪"
                )

            else:

                improvement_msg = (
                    "No change from last session."
                )

        scores = [
            s["score"]
            for s in history
        ]

        # -------------------------------------------------
        # FUTURE SCORE PREDICTION
        # -------------------------------------------------

        prediction = ""

        if len(history) >= 4:

            X = np.array(
                range(len(scores))
            ).reshape(-1, 1)

            y = np.array(scores)

            trend_model = LinearRegression()

            trend_model.fit(X, y)

            next_session = np.array([
                [len(scores)]
            ])

            future_score = trend_model.predict(
                next_session
            )[0]

            future_score = min(
                100,
                future_score
            )

            prediction = (
                f"Projected next-session score: "
                f"{future_score:.1f}"
            )

        # -------------------------------------------------
        # FINAL PERFORMANCE RESULTS
        # -------------------------------------------------

        level = get_level(avg_score)

        tips = list(
            dict.fromkeys(all_tips)
        )

        coach_comment = str(
            generate_ai_comment(
                avg_score,
                angles,
                speed
            )
        )

        comparison = str(
            athlete_comparison(avg_score)
        )

        motivation = str(
            motivation_line()
        )

        # -------------------------------------------------
        # WEAKNESS ANALYSIS
        # -------------------------------------------------

        weakness = []

        technique = technique_score

        speed = speed_score

        control = balance_score

        stability = balance_score

        if technique < 70:

            weakness.append(
                "Technique needs improvement"
            )

        if speed < 70:

            weakness.append(
                "Movement speed is below optimal level"
            )

        if control < 70:

            weakness.append(
                "Ball/body control needs improvement"
            )

        if stability < 70:

            weakness.append(
                "Body balance and stability should be improved"
            )

        if not weakness:

            weakness.append(
                "No major weaknesses detected. Excellent performance."
            )

        # -------------------------------------------------
        # MEDAL SYSTEM
        # -------------------------------------------------

        if avg_score >= 85:

            medal = "Gold 🥇"

        elif avg_score >= 70:

            medal = "Silver 🥈"

        elif avg_score >= 55:

            medal = "Bronze 🥉"

        else:

            medal = "Developing 💪"

        # -------------------------------------------------
        # GRADE SYSTEM
        # -------------------------------------------------

        def get_grade(score):

            if score >= 85:
                return "A+"

            if score >= 70:
                return "A"

            if score >= 55:
                return "B"

            if score >= 40:
                return "C"

            return "D"

        grade = get_grade(avg_score)

        # -------------------------------------------------
        # PDF REPORT
        # -------------------------------------------------

        logger.info(
            "[%s] GENERATING PDF REPORT",
            request_id
        )

        create_pdf_report(

            pdf_path=pdf_path,

            player_name=player_name,

            avg_score=avg_score,

            level=level,

            grade=grade,

            technique_score=technique_score,

            balance_score=balance_score,

            speed_score=speed_score,

            comparison=comparison,

            tips=tips,

            coach_comment=coach_comment,

            motivation=motivation

        )

        logger.info(
            "[%s] VIDEO ANALYSIS COMPLETED SUCCESSFULLY",
            request_id
        )

        # -------------------------------------------------
        # API RESPONSE
        # -------------------------------------------------

        return {

            "status": "analyzed",

            "ai_insight": ai_insight(scores),

            "training_plan": training_plan(avg_score),

            "history": history,

            "improvement": improvement_msg,

            "prediction": prediction,

            "medal": medal,

            "leaderboard": leaderboard,

            # PERFORMANCE SCORES

            "avg_score": float(avg_score),

            "technique": float(technique),

            "balance": float(balance),

            "speed_score": float(speed_score),

            # SKILL LEVEL AND FEEDBACK

            "skill_level": level,

            "tips": tips,

            "weakness": weakness,

            # OUTPUT FILES

            "output_video": f"/outputs/converted_{name}.mp4",

            "pdf_report": f"/outputs/report_{name}.pdf"

        }

    # -------------------------------------------------
    # ERROR HANDLING
    # -------------------------------------------------

    except HTTPException:

        logger.warning(
            "[%s] REQUEST ENDED WITH HTTP ERROR",
            request_id
        )

        raise

    except subprocess.TimeoutExpired:

        logger.exception(
            "[%s] FFMPEG TIMED OUT",
            request_id
        )

        raise HTTPException(
            status_code=504,
            detail=(
                "Video conversion took too long. "
                "Please try a shorter video."
            )
        )

    except Exception as e:

        logger.exception(
            "[%s] VIDEO ANALYSIS FAILED",
            request_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Video analysis failed: "
                f"{type(e).__name__}: {str(e)}"
            )
        )

    finally:

        if cap is not None:
            cap.release()

        if out is not None:
            out.release()

        try:
            file.file.close()
        except Exception:
            pass
