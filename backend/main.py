from fastapi import FastAPI, UploadFile, File
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
from fastapi import Form
import json
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression





app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PATHS ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR,"..","models","skill_model.pkl")
MODEL_PATH = os.path.join(BASE_DIR,"..","models","yolov8s-pose.onnx")

UPLOAD_FOLDER = os.path.join(BASE_DIR,"..","videos")
OUTPUT_FOLDER = os.path.join(BASE_DIR,"..","outputs")

os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs(OUTPUT_FOLDER,exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

# ---------------- LOAD MODELS ---------------- #

skill_model = joblib.load(MODEL_FILE)
session = ort.InferenceSession(MODEL_PATH)

prev_center = None

PLAYERS_FILE = "players.json"

def load_players():
    if not os.path.exists("players.json"):
        return {}

    try:
        with open("players.json","r") as f:
            return json.load(f)
    except:
        return {}


def save_players(data):
    with open(PLAYERS_FILE,"w") as f:
        json.dump(data,f,indent=4)



# ---------------- POSE ---------------- #

def run_pose(frame):
    img = cv2.resize(frame,(640,640))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img,0)

    return session.run(None,{"images":img})

def extract_keypoints(outputs, frame_shape):
    h,w,_ = frame_shape

    preds = outputs[0]
    preds = np.squeeze(preds).T

    if len(preds)==0:
        return []

    best = preds[np.argmax(preds[:,4])]

    if best[4] < 0.3:
        return []

    kpts = best[5:56].reshape(17,3)

    keypoints=[]
    for x,y,conf in kpts:
        if conf>0.5:
            # SCALE from 640x640 → original frame
            px = int(x/640 * w)
            py = int(y/640 * h)
            keypoints.append((px,py))
        else:
            keypoints.append((0,0))

    return keypoints

# ---------------- SKELETON ---------------- #

SKELETON=[
    (5,7),(7,9),
    (6,8),(8,10),
    (5,6),
    (5,11),(6,12),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16)
]

def draw_skeleton(frame,kpts):
    for x,y in kpts:
        if x>0 and y>0:
            cv2.circle(frame,(x,y),5,(0,255,0),-1)

    for p1,p2 in SKELETON:
        if (
            p1<len(kpts) and p2<len(kpts)
            and kpts[p1]!=(0,0)
            and kpts[p2]!=(0,0)
        ):
            cv2.line(frame,kpts[p1],kpts[p2],(255,0,0),2)

    return frame

# ---------------- ANGLES ---------------- #

def calculate_angle(a,b,c):
    a,b,c=np.array(a),np.array(b),np.array(c)
    ba=a-b
    bc=c-b
    cosine=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(cosine))

def extract_angles(k):
    try:
        return {
            "right_elbow":calculate_angle(k[6],k[8],k[10]),
            "left_elbow":calculate_angle(k[5],k[7],k[9]),
            "right_knee":calculate_angle(k[12],k[14],k[16]),
            "left_knee":calculate_angle(k[11],k[13],k[15])
        }
    except:
        return None

def compute_speed(k):
    global prev_center
    hips=[k[11],k[12]]

    if (0,0) in hips:
        return 0

    center=np.mean(hips,axis=0)

    if prev_center is None:
        prev_center=center
        return 0

    speed=np.linalg.norm(center-prev_center)
    prev_center=center
    return speed

# ---------------- LEVEL ---------------- #

def get_level(score):
    if score<50: return "Beginner"
    elif score<75: return "Intermediate"
    else: return "Advanced"

# ---------------- API ---------------- #




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

    # Pick base comment
    if score > 80:
        base = random.choice(elite_comments)
    elif score > 60:
        base = random.choice(good_comments)
    elif score > 40:
        base = random.choice(avg_comments)
    else:
        base = random.choice(beginner_comments)

    # Add personalized tip feedback
    if tips:
        tip_line = f" Key focus: {tips[0]}."
    else:
        tip_line = ""

    return base + tip_line

def generate_ai_comment(score, angles, speed):

    comments = []

    # Technique feedback
    if angles:
        if angles["right_elbow"] < 45:
            comments.append("Your arm extension needs improvement for stronger strokes.")
        else:
            comments.append("Great elbow control — your technique looks refined.")

    # Speed feedback
    if speed < 5:
        comments.append("Try increasing your movement speed to improve performance.")
    else:
        comments.append("Nice pace and movement — that gives you an edge.")

    # Score feedback
    if score > 80:
        comments.append("You’re performing at a competitive athlete level.")
    elif score > 60:
        comments.append("Solid performance, but there’s room to level up.")
    else:
        comments.append("You’re building the basics — keep training consistently.")

    return " ".join(comments)

def athlete_comparison(score):

    if score > 85:
        return "Your movement quality is approaching professional athlete standards."
    elif score > 70:
        return "Your form resembles trained club-level players."
    else:
        return "You’re on the path — consistent practice will elevate you."

def motivation_line():

    lines = [
        "Champions train when nobody is watching.",
        "Small improvements daily create elite athletes.",
        "Focus on progress, not perfection.",
        "Your only competition is yesterday’s version of you.",
        "Consistency beats intensity."
    ]

    return random.choice(lines)

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


def training_plan(score):
    if score < 50:
        return "Focus on basic form drills."
    elif score < 75:
        return "Add agility and reaction drills."
    else:
        return "Maintain elite-level training."


def get_coach_comment(score):
    if score > 80:
        return "Elite performance. You show strong control and confidence."
    elif score > 60:
        return "Good work. Focus on consistency and speed."
    elif score > 40:
        return "Average form. Improve technique and posture."
    else:
        return "Beginner level. Train regularly to build fundamentals."


@app.post("/upload-video/")
async def upload_video(
    file: UploadFile = File(...),
    player_name: str = Form(...)
):

    global prev_center
    prev_center=None

    file_path=os.path.join(UPLOAD_FOLDER,file.filename)

    with open(file_path,"wb") as buf:
        shutil.copyfileobj(file.file,buf)

    cap=cv2.VideoCapture(file_path)

    w=int(cap.get(3))
    h=int(cap.get(4))
    fps=int(cap.get(5)) or 25

    name=os.path.splitext(file.filename)[0]

    out_path=os.path.join(OUTPUT_FOLDER,f"annotated_{name}.mp4")

    out=cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,(w,h)
    )



    all_scores = []
    all_tips = []

    tech_list = []
    bal_list = []
    speed_list = []

    # ---------------- LOOP ----------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputs = run_pose(frame)
        kpts = extract_keypoints(outputs, frame.shape)

        if len(kpts) == 0:
            out.write(frame)
            continue

        angles = extract_angles(kpts)

        if angles:
            speed = compute_speed(kpts)

            # ✅ MODEL INPUT (DO NOT CHANGE)
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

            prediction = skill_model.predict(df)[0]
            all_scores.append(float(prediction))

            # ✅ ANALYTICS (FOR DASHBOARD)
            technique = np.mean([
                angles["right_elbow"],
                angles["left_elbow"]
            ])

            technique_score = max(0, min(100, technique))

            balance = np.mean([
                angles["right_knee"],
                angles["left_knee"]
            ])

            balance_score = max(0, min(100, balance))

            speed_score = min(speed * 15, 100)

            tech_list.append(technique)
            bal_list.append(balance)
            speed_list.append(speed_score)

            final_score = (
                    technique_score * 0.4 +
                    balance_score * 0.35 +
                    speed_score * 0.25
            )
            final_score = (final_score + prediction) / 2

            # ✅ TIPS
            # ---------------- SMART FEEDBACK ----------------

            # ELBOW TECHNIQUE
            if angles["right_elbow"] < 45:
                all_tips.append(
                    "Your right elbow angle is too closed. This reduces power generation. "
                    "Try extending your arm fully during execution for stronger shots."
                )

            if angles["left_elbow"] < 45:
                all_tips.append(
                    "Left elbow positioning is limiting your swing range. "
                    "Keep your arm more open to improve control and balance."
                )

            # KNEE / BALANCE
            if angles["right_knee"] < 140:
                all_tips.append(
                    "Your right knee is not bending enough. "
                    "Proper knee bend helps in shock absorption and stability."
                )

            if angles["left_knee"] < 140:
                all_tips.append(
                    "Left knee mechanics need improvement. "
                    "A deeper bend improves agility and quick direction changes."
                )

            # SPEED
            if speed < 3:
                all_tips.append(
                    "Your movement speed is low. Faster footwork increases reaction time "
                    "and competitive performance."
                )

            # POSTURE
            if technique < 60 and balance < 60:
                all_tips.append(
                    "Your overall posture needs work. Focus on body alignment and core "
                    "stability during play."
                )

        frame = draw_skeleton(frame, kpts)
        out.write(frame)

    cap.release()
    out.release()

    # -------- FFMPEG -------- #

    converted_path=os.path.join(OUTPUT_FOLDER,f"converted_{name}.mp4")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i",out_path,
        "-vcodec","libx264",
        "-pix_fmt","yuv420p",
        converted_path
    ])

    avg_score=sum(all_scores)/len(all_scores) if all_scores else 0
    players = load_players()

    if player_name not in players:
        players[player_name] = []

    players[player_name].append({
        "date": dt.now().strftime("%d %b %Y"),
        "score": avg_score
    })

    save_players(players)

    weakness = []

    # Example metrics derived from score
    technique = avg_score
    speed = avg_score * 0.9
    control = avg_score * 0.85
    stability = avg_score * 0.8

    if technique < 70:
        weakness.append("Technique needs improvement")

    if speed < 70:
        weakness.append("Movement speed is below optimal level")

    if control < 70:
        weakness.append("Ball/body control needs improvement")

    if stability < 70:
        weakness.append("Body balance and stability should be improved")

    if not weakness:
        weakness.append("No major weaknesses detected. Excellent performance.")

    # -------- LEADERBOARD --------

    leaderboard = sorted(
        [
            {"name": k, "score": v[-1]["score"]}
            for k, v in players.items() if v
        ],
        key=lambda x: x["score"],
        reverse=True
    )[:5]

    history = players[player_name]
    improvement_msg = ""

    if len(history) >= 2:
        last = history[-1]["score"]
        prev = history[-2]["score"]

        diff = last - prev
        percent = (diff / prev) * 100 if prev > 0 else 0

        if diff > 0:
            improvement_msg = f"{player_name} improved {percent:.1f}% since last session 🔥"
        elif diff < 0:
            improvement_msg = f"{player_name} dropped {abs(percent):.1f}% — time to train harder 💪"
        else:
            improvement_msg = "No change from last session."
    scores = [s["score"] for s in history]

    prediction = ""

    if len(history) >= 4:
        scores = [h["score"] for h in history]

        # X = session numbers
        X = np.array(range(len(scores))).reshape(-1, 1)
        y = np.array(scores)

        model = LinearRegression()
        model.fit(X, y)

        # predict next session
        next_session = np.array([[len(scores)]])
        future_score = model.predict(next_session)[0]

        future_score = min(100, future_score)

        prediction = f"Projected next-session score: {future_score:.1f}"



    level=get_level(avg_score)
    tips=list(set(all_tips))
    coach_comment = str(generate_ai_comment(avg_score, angles, speed))
    comparison = str(athlete_comparison(avg_score))
    motivation = str(motivation_line())

    # -------- MEDAL SYSTEM --------
    if avg_score >= 85:
        medal = "Gold 🥇"
    elif avg_score >= 70:
        medal = "Silver 🥈"
    elif avg_score >= 55:
        medal = "Bronze 🥉"
    else:
        medal = "Developing 💪"

    # -------- PDF -------- #

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor


    pdf_path=os.path.join(OUTPUT_FOLDER,f"report_{name}.pdf")


    c = canvas.Canvas(pdf_path, pagesize=A4)

    # COLORS
    dark = HexColor("#111827")
    green = HexColor("#22c55e")
    grey = HexColor("#374151")

    # HEADER BAR
    c.setFillColor(dark)
    c.rect(0, 770, 600, 70, fill=1)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(140, 800, "AI SPORTS PERFORMANCE REPORT")

    # PLAYER INFO
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 12)

    date = dt.now().strftime("%d %b %Y")

    c.drawString(50, 740, f"Player Name: {player_name}")
    c.drawString(50, 720, f"Date: {date}")

    # SCORE BOX
    c.setFillColor(green)
    c.rect(50, 650, 500, 50, fill=1)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(200, 670, f"Skill Score: {avg_score:.2f}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 580, "Performance Breakdown")

    c.setFont("Helvetica", 12)
    c.drawString(70, 560, f"Technique: {technique_score:.1f}")
    c.drawString(70, 540, f"Balance: {balance_score:.1f}")
    c.drawString(70, 520, f"Speed: {speed_score:.1f}")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 480, "AI Insights")

    insight = (
        "Your movement pattern shows good coordination. "
        "Improving joint extension and explosive movement "
        "can elevate performance to competitive level."
    )

    text = c.beginText(60, 455)
    text.setFont("Helvetica", 12)
    text.textLine(insight)
    c.drawText(text)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 420, "Athlete Comparison")

    c.setFont("Helvetica-Oblique", 12)
    c.drawString(60, 400, comparison)

    c.setFillColor(HexColor("#e5e7eb"))




    # LEVEL + GRADE
    def get_grade(score):
        if score >= 85: return "A+"
        if score >= 70: return "A"
        if score >= 55: return "B"
        if score >= 40: return "C"
        return "D"

    grade = get_grade(avg_score)

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)

    c.drawString(50, 610, f"Skill Level: {level}")
    c.drawString(350, 610, f"Grade: {grade}")



    # ---------- TEXT WRAP ----------
    def draw_wrapped_text(text, x, y, max_chars=90):
        text_obj = c.beginText(x, y)
        text_obj.setFont("Helvetica", 12)

        words = text.split()
        line = ""

        for word in words:
            if len(line + word) <= max_chars:
                line += word + " "
            else:
                text_obj.textLine(line)
                line = word + " "

        text_obj.textLine(line)
        c.drawText(text_obj)

        return text_obj.getY()

    # ---------- PERFORMANCE ----------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 560, "Performance Insights")

    y = 535

    c.setFont("Helvetica", 12)

    for t in tips:
        if y < 380:  # STOP before coach zone
            break

        y = draw_wrapped_text("• " + t, 70, y)
        y -= 8

    # ---------- COACH FEEDBACK ----------
    coach_y = 345

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, coach_y, "Coach Feedback")



    c.setFont("Helvetica-Oblique", 12)
    y = draw_wrapped_text(coach_comment, 60, coach_y -25)

    # ---------- MOTIVATION (ONLY ONCE) ----------
    mot_y = 210

    c.setFillColor(HexColor("#e8fdf5"))
    c.rect(50, mot_y - 20, 500, 45, fill=1)

    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, mot_y + 8, "Motivation")

    c.setFont("Helvetica-Oblique", 12)
    c.drawString(60, mot_y - 8, motivation)

    # ---------------- FOOTER ----------------

    c.setFillColor(grey)
    c.rect(0, 0, 600, 40, fill=1)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica", 10)
    c.drawString(180, 15, "Generated by Real-Time MultiSport Skill Estimator")

    c.save()

    return {
        "status": "analyzed",
        "ai_insight": ai_insight(scores),
        "training_plan": training_plan(avg_score),
        "history": history,
        "improvement": improvement_msg,
        "prediction": prediction,
        "medal": medal,
        "leaderboard": leaderboard,

        # ⭐ Scores
        "avg_score": float(avg_score),
        "technique": float(technique),
        "balance": float(balance),
        "speed_score": float(speed_score),

        # ⭐ Level & tips
        "skill_level": level,
        "tips": tips,
        "weakness": weakness,

        # ⭐ Files
        "output_video": f"/outputs/converted_{name}.mp4",
        "pdf_report": f"/outputs/report_{name}.pdf"
    }