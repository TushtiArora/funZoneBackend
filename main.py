from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_cors import CORS
import cv2
import time
import os
import handTrackingModule as htm
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)
widCam, htCam = 640, 480

camera = cv2.VideoCapture(0)
camera.set(3, widCam)
camera.set(4, htCam)

latest_review_frame = None

def gen_frames():
    detector = htm.handDetector(detectionCon=0.75)
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=False)
        
        if len(lmList) != 0:
            tipIds = [4, 8, 12, 16, 20]
            fingers = []
            # Thumb
            if lmList[4][1] > lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            totalFingers = sum(fingers)
            cv2.putText(frame, f'Rating: {totalFingers}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "Flask Backend is running"

@app.route('/rate', methods=['POST'])
def rate():
    print('Received request to /rate')
    if 'image' not in request.files:
        print('No image in request.files')
        return jsonify({'status': 'error', 'message': 'No image provided'})
    
    file = request.files['image']
    print(f'Received file: {file.filename}')
    # Convert to OpenCV format
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        print('cv2.imdecode failed, frame is None')
        return jsonify({'status': 'error', 'message': 'Invalid image'})
    
    detector = htm.handDetector(detectionCon=0.75)
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    print(f'Landmarks found: {lmList}')
    
    if not lmList:
        print('No hand detected in image')
        return jsonify({'status': 'error', 'message': 'No hand detected'})
    
    tipIds = [4, 8, 12, 16, 20]
    fingers = []
    # Thumb
    if lmList[4][1] > lmList[3][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    # 4 Fingers
    for id in range(1, 5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    totalFingers = sum(fingers)
    print(f'Fingers detected: {fingers}, Total: {totalFingers}')
    
    return jsonify({'status': 'success', 'rating': totalFingers})

@app.route('/submit_review_image', methods=['POST'])
def submit_review_image():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'})
    
    file = request.files['image']
    remark = request.form.get('remark', '')
    
    # Create directory if it doesn't exist
    os.makedirs('capturedFrames', exist_ok=True)
    
    # Save the image with timestamp
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'review_{timestamp}.jpg'
    filepath = os.path.join('capturedFrames', filename)
    file.save(filepath)
    
    return jsonify({
        'status': 'success',
        'message': 'Review submitted successfully',
        'filename': filename
    })

@app.route('/video')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    os.makedirs('capturedFrames', exist_ok=True)
    app.run(debug=True, port=5000)
