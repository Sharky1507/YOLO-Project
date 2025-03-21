# YOLO-Object-Detection Proof of Concept(PoC)

# Flask YOLO Chatbot

## Overview
This is a proof-of-concept Flask application that integrates:
- **YOLOv8** for real-time object detection from a webcam feed.
- **Google Gemini AI** for an interactive chatbot.

## Features
- Streams real-time webcam feed with YOLO object detection.
- Allows users to chat with Gemini AI.
- AI responses consider detected objects in the environment.

## Installation
### Prerequisites
- Python 3.8+
- Flask
- OpenCV
- Ultralytics YOLO
- Google Generative AI SDK

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/flask-yolo-chatbot.git
cd flask-yolo-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Configuration
Replace `your-gemini-api-key` in the code with your actual **Google Gemini API key**.

## Running the App
```bash
python app.py
```
The app will be available at `http://127.0.0.1:5000/`.

## Code Explanation
### 1. Flask App Setup
```python
app = Flask(__name__)
camera = cv2.VideoCapture(0)
```
- Initializes Flask and the webcam.

### 2. YOLO Object Detection
```python
def generate_frames():
    global last_detected_objects
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
```
- Captures webcam frames, detects objects, and overlays results.

### 3. AI Chatbot
```python
@app.route('/chat', methods=['POST'])
def chat():
    global chat_history, last_detected_objects
    user_message = request.json.get("message", "")
    chat_history.append({"role": "user", "content": user_message})
    
    try:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])
        full_prompt = f"Context: Detected objects: {', '.join(last_detected_objects) if last_detected_objects else 'none'}\n\nConversation:\n{history_text}\n\nNew Message: {user_message}\nProvide a helpful response:"
        
        response = gemini_model.generate_content(full_prompt)
        ai_response = response.text
    except Exception as e:
        ai_response = f"Error: {str(e)}"
    
    chat_history.append({"role": "assistant", "content": ai_response})
    return jsonify({"response": ai_response})
```
- Receives user input, passes it to Gemini AI, and returns a response.

## API Endpoints
| Endpoint       | Method | Description |
|---------------|--------|-------------|
| `/`           | GET    | Renders the main webpage |
| `/video_feed` | GET    | Streams real-time video with object detection |
| `/chat`       | POST   | Processes user chat messages with AI |


