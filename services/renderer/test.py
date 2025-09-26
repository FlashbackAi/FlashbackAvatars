import requests
import base64
from PIL import Image
import numpy as np
import cv2

# Load video frames
video_path = "../../avatar_input/vinayone.jpg"
cap = cv2.VideoCapture(video_path)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

cap.release()
print(f"Loaded {len(frames)} frames from video")

# Convert to numpy array: (frames, height, width, channels)
video_array = np.array(frames)

# Test with sample audio (you can use any WAV file)
audio_file = "../../vinay_audio.wav"

# Send to renderer
response = requests.post("http://localhost:9000/render",
                        json={
                            "video": video_array.tolist(),
                            "audio_file": audio_file
                        })

# Debug: Check response
print(f"Response status: {response.status_code}")
print(f"Response content: {response.text}")

# Get animated frame
if response.status_code == 200:
    response_data = response.json()
    print(f"Response keys: {response_data.keys()}")
    if "frame" in response_data:
        animated_frame = response_data["frame"]
        print(f"✅ Got animated frame! Status: {response_data.get('status')}")
        print(f"📝 Message: {response_data.get('message')}")
        
        # Decode and save the result
        frame_data = base64.b64decode(animated_frame)
        with open("generated_frame.png", "wb") as f:
            f.write(frame_data)
        print("💾 Saved generated frame as 'generated_frame.png'")
    else:
        print("❌ No 'frame' key in response")
        print(f"Response: {response_data}")
else:
    print(f"❌ Server error: {response.status_code}")
    print(f"Error details: {response.text}")