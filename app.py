from flask import Flask, render_template, Response, request, jsonify
import cv2
import time
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai

app = Flask(__name__)

model = YOLO("yolov8n.pt")

genai.configure(api_key="AIzaSyCvhHTnUry-WUB6C3gTBZpJSHcP5qN7P04")  
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

camera = None  
last_detected_objects = []  
chat_history = []  

def generate_frames():
    global camera, last_detected_objects

    camera = cv2.VideoCapture(0) 
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        detected_objects = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            obj_name = model.names[cls_id]
            detected_objects.append(obj_name)
        
        last_detected_objects = list(set(detected_objects))  
        
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history, last_detected_objects

    user_message = request.json.get("message", "")
    chat_history.append({"role": "user", "content": user_message})
    
    try:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])
        full_prompt = f"""Context: You're an AI room assistant. Detected objects: {', '.join(last_detected_objects) if last_detected_objects else 'none'}
        
        Conversation History:
        {history_text}
        
        New Message: {user_message}
        
        Provide a helpful, friendly response:"""
        
        response = gemini_model.generate_content(full_prompt)
        ai_response = response.text
    except Exception as e:
        ai_response = f"⚠️ Error generating response: {str(e)}"

    chat_history.append({"role": "assistant", "content": ai_response})
    
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run(debug=True)
