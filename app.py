from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from analyzer import analyze_frame, get_emotion_history

app = Flask(__name__)

camera = cv2.VideoCapture(0)
latest_result = {
    'success': False,
    'dominant_emotion': 'neutral',
    'emotions': {},
    'depression_score': 0,
    'risk_level': 'Analyzing...',
    'risk_color': 'gray',
    'risk_message': 'Camera ke saamne aao...'
}
lock = threading.Lock()

def generate_frames():
    global latest_result
    while True:
        success, frame = camera.read()
        if not success:
            break

        result = analyze_frame(frame)
        with lock:
            latest_result = result

        color_map = {
            'green':  (0, 255, 0),
            'orange': (0, 165, 255),
            'red':    (0, 0, 255),
            'gray':   (128, 128, 128)
        }
        color = color_map.get(result.get('risk_color', 'gray'), (128, 128, 128))

        emotion_text = "Emotion: " + str(result.get('dominant_emotion', ''))
        risk_text = "Risk: " + str(result.get('risk_level', ''))

        cv2.putText(frame, emotion_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, risk_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_analysis')
def get_analysis():
    with lock:
        return jsonify(latest_result)

@app.route('/get_history')
def get_history():
    return jsonify(get_emotion_history())

if __name__ == '__main__':
    app.run(debug=True)