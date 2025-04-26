#main.py
'''
# Smart Risk Detector Flask App
# This Flask app provides endpoints for posture detection using YOLOv8 and pose estimation.
# It also integrates with the Gemini API for generating posture reports.
'''
from dotenv import load_dotenv
load_dotenv()

import os
import base64
import time
import cv2
import numpy as np
import requests
import urllib.parse
import json
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai

# Import your custom modules
from yolo_detector import YOLODetector
from pose_estimator import PoseEstimator
from anomaly_detector import AnomalyDetector

########################################
# 1) Flask App Setup
########################################

app = Flask(__name__, static_folder='static')

# Instantiate YOLO, Pose, and Anomaly modules
detector = YOLODetector(
    model_path="yolov8n.pt",
    device="auto",
    conf_thres=0.7,
    iou_thres=0.45,
    gamma=1.0,
    advanced_box_filter=True
)

pose_estimator = PoseEstimator()

# Instantiate your anomaly detector using the expected argument names.
anomaly_detector = AnomalyDetector(
    body_angle_threshold=55, 
    neck_angle_threshold=50,
    history_window=10,
    deviation_angle_threshold=15
)

########################################
# 2) Helper Functions
########################################

def scale_keypoints(kp_normalized, shape):
    """
    Convert normalized keypoints (range 0..1) to absolute pixel coordinates.
    """
    h, w, _ = shape
    return [(x * w, y * h) for x, y in kp_normalized]

def run_posture_detection(frame):
    """
    Run YOLO detection, pose estimation, convert keypoints to absolute values,
    and then check for deviation if a baseline is set.
    
    Returns:
       (annotated_frame, posture_status, angles)
    """
    # 1) Detect persons using YOLO.
    boxes, confs = detector.detect(frame)
    if not boxes:
        return frame, "no_person", {}

    # 2) Choose the detection with highest confidence.
    max_conf_idx = max(range(len(confs)), key=lambda i: confs[i])
    # (x1, y1, x2, y2) = boxes[max_conf_idx]  # Not used further in this example

    # 3) Estimate the pose.
    kp_norm, annotated_frame = pose_estimator.get_pose(frame)
    kp_abs = scale_keypoints(kp_norm, frame.shape)

    # 4) Compute the main angles using the anomaly detector helper.
    angles = anomaly_detector._compute_angles(kp_abs)
    
    # 5) Determine posture status:
    # If a baseline exists, compare against it.
    if anomaly_detector.has_baseline():
        is_deviated = anomaly_detector.is_deviated_from_baseline(kp_abs)
        posture_status = "bad" if is_deviated else "good"
    else:
        # Fallback: use the "fall_like" detection.
        is_bad = anomaly_detector.is_fall_like(kp_abs)
        posture_status = "bad" if is_bad else "good"

    return annotated_frame, posture_status, angles

########################################
# 3) Baseline Capture Endpoint
########################################

@app.route("/capture_baseline", methods=["POST"])
def capture_baseline():
    """
    Expects a JSON payload with:
      { "image": "data:image/jpeg;base64,..." }
    Processes the image to obtain keypoints and sets the baseline posture.
    Returns the computed baseline angles.
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No 'image' field in request"}), 400

    try:
        header, b64_data = data["image"].split(",", 1)
    except ValueError:
        return jsonify({"error": "Invalid image format"}), 400

    img_bytes = base64.b64decode(b64_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Estimate pose and get keypoints
    kp_norm, annotated_frame = pose_estimator.get_pose(frame)
    kp_abs = scale_keypoints(kp_norm, frame.shape)

    if len(kp_abs) < 15:
        return jsonify({"success": False, "error": "Not enough keypoints to set baseline."})

    # Set the baseline posture in the anomaly detector.
    anomaly_detector.set_baseline(kp_abs)
    baseline_angles = anomaly_detector._compute_angles(kp_abs)

    return jsonify({
        "success": True,
        "baseline": kp_abs,
        "angles": baseline_angles
    })

########################################
# 3) API Functions
########################################

def get_access_token(api_key, secret_key):
    """获取百度API访问令牌"""
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key}
    response = requests.post(url, params=params, proxies={"http": None, "https": None})
    token = response.json().get("access_token")
    return token

def generate_posture_advice(token, posture_status, angles):
    """使用百度文心一言生成姿势建议"""
    if token is None:
        return "未能连接到AI服务，请检查您的姿势，确保背部挺直，颈部保持自然角度，双肩放松。"
    
    llm_url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-4.0-turbo-128k?access_token={token}"
    
    user_prompt = (
        f"你是一位关心办公人员健康的虚拟助手，语气友好且个性化。请根据以下姿势数据，给出一条简短、"
        f"有效且容易理解的改进建议(70-100字)：\n\n"
        f"姿势状态：{posture_status}\n"
        f"颈部角度：{angles.get('neck', '--')}度\n"
        f"左侧身体角度：{angles.get('left_body', '--')}度\n"
        f"右侧身体角度：{angles.get('right_body', '--')}度\n"
        f"平均身体角度：{angles.get('avg_body', '--')}度\n\n"
        f"重要提示：根据度数实时给出建议，不要播报度数和数据，而是根据数据智能化但是专业的提出建议。"
    )
    
    payload = json.dumps({
        "messages": [{"role": "user", "content": user_prompt}],
        "penalty_score": 1,
        "disable_search": True
    }, ensure_ascii=False)
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(llm_url, headers=headers, data=payload.encode("utf-8"), proxies={"http": None, "https": None})
    
    try:
        result = response.json()
        return result.get('result', "未能获取个性化建议，请注意您的姿势，保持背部挺直，颈部放松。")
    except:
        return "系统暂时无法生成建议，请尝试保持良好坐姿，每小时起身活动5-10分钟。"

def generate_audio(token, text):
    """使用百度TTS API生成音频"""
    if not token:
        return None
        
    tts_url = "http://tsn.baidu.com/text2audio"
    encoded_text = urllib.parse.quote_plus(text)
    
    data = {
        "tex": encoded_text,
        "lan": "zh",
        "cuid": "posepilot",
        "ctp": "1",
        "aue": "3",
        "tok": token
    }
    
    response = requests.post(tts_url, data=data, proxies={"http": None, "https": None})
    
    if response.headers.get("Content-Type", "").startswith("audio/"):
        timestamp = int(time.time())
        audio_dir = os.path.join(app.static_folder, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        audio_file = f"report_{timestamp}.mp3"
        audio_path = os.path.join(audio_dir, audio_file)
        
        with open(audio_path, "wb") as f:
            f.write(response.content)
        
        return f"/static/audio/{audio_file}"
    return None

def get_posture_report(angles, posture_status):
    """整合姿势分析，生成文本建议和语音报告"""
    try:
        # Baidu API credentials
        llm_api_key = "2YcoX4HqbcA6pMEolmNknwTQ"
        llm_secret_key = "3eeNrmrpVcKnBEes3MZrcQJeMxqfLhEH"
        tts_api_key = "UONsI3GbC0ABHkuWS2b8coxG" 
        tts_secret_key = "FpvrKuHJjVJrf1K6HEuhM7wtXsTqFO6K"
        
        # 获取LLM令牌
        llm_token = get_access_token(llm_api_key, llm_secret_key)
        ai_advice = generate_posture_advice(llm_token, posture_status, angles)
        
        # 获取TTS令牌并生成音频
        tts_token = get_access_token(tts_api_key, tts_secret_key)
        audio_url = generate_audio(tts_token, ai_advice) or ""
        
        return {
            "audio_url": audio_url,
            "text": ai_advice,
            "ai_advice": ai_advice
        }
    except Exception as e:
        return {"error": f"Error calling API: {e}", "text": "无法生成姿势报告"}

########################################
# 4) Posture Detection Endpoint
########################################

# 添加性能监控
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.3f} seconds")
        return result
    return wrapper

@app.route("/detect_posture", methods=["POST"])
@timeit
def detect_posture():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No 'image' field in request"}), 400

    try:
        # 优化图像解码
        header, b64_data = data["image"].split(",", 1)
        img_bytes = base64.b64decode(b64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        
        # 使用更快的解码方式
        frame = cv2.imdecode(np_arr, cv2.IMREAD_REDUCED_COLOR_2)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        # 优化YOLO检测
        boxes, confs = detector.detect(frame)
        if not boxes:
            return jsonify({"posture": "no_person", "annotated_image": data["image"]})

    except ValueError:
        return jsonify({"error": "Invalid image format"}), 400

    img_bytes = base64.b64decode(b64_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    annotated_frame, posture_status, angles = run_posture_detection(frame)

    # Encode the annotated frame back to a base64 URL
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")
    annotated_b64_url = f"data:image/jpeg;base64,{annotated_b64}"

    # Generate a report using Gemini API if posture is bad
    report_data = {"text": "", "audio_url": ""}
    if posture_status == "bad":
        report_data = get_posture_report(angles, posture_status)

    return jsonify({
        "posture": posture_status,
        "annotated_image": annotated_b64_url,
        "angles": angles,
        "report": report_data.get("text", ""),
        "audio_url": report_data.get("audio_url", "")
    })

@app.route("/test_detection", methods=["POST"])
def test_detection():
    """
    測試姿勢檢測功能
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No 'image' field in request"}), 400

    try:
        header, b64_data = data["image"].split(",", 1)
    except ValueError:
        return jsonify({"error": "Invalid image format"}), 400

    img_bytes = base64.b64decode(b64_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    # 測試姿勢檢測
    annotated_frame, posture_status, angles = run_posture_detection(frame)

    # 返回結果
    return jsonify({
        "posture": posture_status,
        "angles": angles
    })

@app.route("/test_voice", methods=["POST"])
def test_voice():
    """
    測試語音播報功能
    """
    # 移除語音播報測試
    return jsonify({"success": True})

########################################
# 5) Serve the Index HTML
########################################

@app.route("/")
def serve_index():
    """
    Serve the index.html file from the 'static' folder.
    """
    return send_from_directory(app.static_folder, "index.html")

########################################
# 6) Run the Flask App
########################################

if __name__ == "__main__":
    # By default, run on localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)