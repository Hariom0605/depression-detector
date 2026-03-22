import cv2
from deepface import DeepFace
import numpy as np
from collections import deque
import time

# Last 60 seconds ka data store karne ke liye
emotion_history = deque(maxlen=60)

# Depression indicators - ye emotions zyada ho toh concern
DEPRESSION_INDICATORS = ['sad', 'fear', 'disgust', 'angry']
POSITIVE_EMOTIONS = ['happy', 'surprise']

def analyze_frame(frame):
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result['emotion']
        dominant = result['dominant_emotion']

        # Score calculate karo
        depression_score = sum(emotions.get(e, 0) for e in DEPRESSION_INDICATORS)
        positive_score = sum(emotions.get(e, 0) for e in POSITIVE_EMOTIONS)

        # History mein add karo
        emotion_history.append({
            'time': time.time(),
            'dominant': dominant,
            'depression_score': depression_score,
            'positive_score': positive_score,
            'emotions': emotions
        })

        # Risk level calculate karo
        risk = get_risk_level()

        return {
            'success': True,
            'dominant_emotion': dominant,
            'emotions': {k: round(v, 1) for k, v in emotions.items()},
            'depression_score': round(depression_score, 1),
            'risk_level': risk['level'],
            'risk_color': risk['color'],
            'risk_message': risk['message']
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'dominant_emotion': 'neutral',
            'emotions': {},
            'depression_score': 0,
            'risk_level': 'Analyzing',
            'risk_color': 'gray',
            'risk_message': 'Camera ke saamne aao...'
        }

def get_risk_level():
    if len(emotion_history) < 5:
        return {
            'level': 'Analyzing...',
            'color': 'gray',
            'message': 'Thoda ruko — analyze kar raha hoon'
        }

    # Average depression score last 30 entries ka
    recent = list(emotion_history)[-30:]
    avg_depression = np.mean([e['depression_score'] for e in recent])
    avg_positive = np.mean([e['positive_score'] for e in recent])

    # Flat affect check - koi bhi emotion zyada nahi
    avg_neutral = np.mean([
        e['emotions'].get('neutral', 0) for e in recent
        if e['emotions']
    ])

    if avg_depression > 60 or avg_neutral > 80:
        return {
            'level': 'High Concern 🔴',
            'color': 'red',
            'message': 'Negative emotions zyada detect ho rahe hain. Kisi se baat karo.'
        }
    elif avg_depression > 35 or avg_neutral > 60:
        return {
            'level': 'Moderate 🟡',
            'color': 'orange',
            'message': 'Thodi stress detect ho rahi hai. Self-care karo.'
        }
    else:
        return {
            'level': 'Low Concern 🟢',
            'color': 'green',
            'message': 'Sab theek lag raha hai! Positive raho.'
        }

def get_emotion_history():
    if not emotion_history:
        return []

    history = list(emotion_history)
    return [{
        'time': round(h['time'] - history[0]['time'], 1),
        'depression_score': round(h['depression_score'], 1),
        'positive_score': round(h['positive_score'], 1),
        'dominant': h['dominant']
    } for h in history]