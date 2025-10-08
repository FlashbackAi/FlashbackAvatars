# External Website Integration Guide

How to integrate Flashback Voice Chat into your own website.

## Server URL

After deployment:
```
http://YOUR_SERVER_IP:8001
```

---

## Option 1: WebSocket Integration (Recommended)

Real-time bidirectional communication.

### JavaScript Client Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Voice Chat Integration</title>
</head>
<body>
    <button id="connect">Connect</button>
    <button id="disconnect">Disconnect</button>
    <input type="text" id="message" placeholder="Type message...">
    <button id="send">Send</button>
    <div id="messages"></div>
    <audio id="audio-player"></audio>

    <script>
        let ws = null;

        // Connect to voice chat server
        document.getElementById('connect').onclick = () => {
            ws = new WebSocket('ws://YOUR_SERVER_IP:8001/ws');

            ws.onopen = () => {
                console.log('Connected to voice chat');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'response') {
                    // Display text response
                    document.getElementById('messages').innerHTML +=
                        `<p><strong>Vinay:</strong> ${data.text}</p>`;

                    // Play voice response
                    if (data.audio_url) {
                        const audioPlayer = document.getElementById('audio-player');
                        audioPlayer.src = `http://YOUR_SERVER_IP:8001${data.audio_url}`;
                        audioPlayer.play();
                    }
                }
            };

            ws.onclose = () => {
                console.log('Disconnected');
            };
        };

        // Disconnect
        document.getElementById('disconnect').onclick = () => {
            if (ws) ws.close();
        };

        // Send message
        document.getElementById('send').onclick = () => {
            const message = document.getElementById('message').value;

            if (ws && message) {
                ws.send(JSON.stringify({
                    type: 'text',
                    message: message
                }));

                // Display user message
                document.getElementById('messages').innerHTML +=
                    `<p><strong>You:</strong> ${message}</p>`;

                document.getElementById('message').value = '';
            }
        };
    </script>
</body>
</html>
```

---

## Option 2: REST API Integration

For non-real-time use cases.

### Text Message Endpoint

**POST** `/api/text-message`

**Request:**
```json
{
    "message": "Hi, what's your name?"
}
```

**Response:**
```json
{
    "type": "response",
    "text": "Good morning. I'm Vinay.",
    "audio_url": "/audio/audio_abc123.wav",
    "confidence": 1.0,
    "model": {
        "kind": "openai",
        "name": "gpt-4o-mini (phatic)"
    },
    "citations": []
}
```

**JavaScript Example:**
```javascript
async function sendTextMessage(message) {
    const response = await fetch('http://YOUR_SERVER_IP:8001/api/text-message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    });

    const data = await response.json();

    // Display text
    console.log('Vinay:', data.text);

    // Play audio
    const audio = new Audio(`http://YOUR_SERVER_IP:8001${data.audio_url}`);
    audio.play();
}

// Usage
sendTextMessage("Hi, what's your name?");
```

---

### Voice Message Endpoint

**POST** `/api/voice-message`

Send audio file, get text + voice response.

**Request:**
- Form-data with audio file
- Field name: `audio`
- Supported formats: WAV, MP3, M4A

**Response:**
```json
{
    "type": "response",
    "text": "Good morning. I'm Vinay.",
    "audio_url": "/audio/audio_abc123.wav",
    "confidence": 1.0
}
```

**JavaScript Example:**
```javascript
async function sendVoiceMessage(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    const response = await fetch('http://YOUR_SERVER_IP:8001/api/voice-message', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();

    // Display text
    console.log('Vinay:', data.text);

    // Play response audio
    const audio = new Audio(`http://YOUR_SERVER_IP:8001${data.audio_url}`);
    audio.play();
}

// Record audio from microphone
async function recordAndSend() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        await sendVoiceMessage(audioBlob);
    };

    // Start recording
    mediaRecorder.start();

    // Stop after 5 seconds (or use button)
    setTimeout(() => {
        mediaRecorder.stop();
    }, 5000);
}
```

---

## Option 3: React Integration

### Custom Hook

```jsx
import { useState, useEffect, useRef } from 'react';

export function useVoiceChat(serverUrl) {
    const [connected, setConnected] = useState(false);
    const [messages, setMessages] = useState([]);
    const wsRef = useRef(null);
    const audioRef = useRef(new Audio());

    const connect = () => {
        const ws = new WebSocket(`${serverUrl}/ws`);

        ws.onopen = () => {
            setConnected(true);
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'response') {
                setMessages(prev => [...prev, {
                    sender: 'vinay',
                    text: data.text
                }]);

                // Play audio
                if (data.audio_url) {
                    audioRef.current.src = `${serverUrl}${data.audio_url}`;
                    audioRef.current.play();
                }
            }
        };

        ws.onclose = () => {
            setConnected(false);
        };

        wsRef.current = ws;
    };

    const disconnect = () => {
        if (wsRef.current) {
            wsRef.current.close();
        }
    };

    const sendMessage = (message) => {
        if (wsRef.current && message) {
            setMessages(prev => [...prev, {
                sender: 'user',
                text: message
            }]);

            wsRef.current.send(JSON.stringify({
                type: 'text',
                message: message
            }));
        }
    };

    useEffect(() => {
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    return {
        connected,
        messages,
        connect,
        disconnect,
        sendMessage
    };
}

// Usage in component
function VoiceChatWidget() {
    const { connected, messages, connect, disconnect, sendMessage } =
        useVoiceChat('ws://YOUR_SERVER_IP:8001');

    const [input, setInput] = useState('');

    return (
        <div>
            {!connected ? (
                <button onClick={connect}>Connect</button>
            ) : (
                <button onClick={disconnect}>Disconnect</button>
            )}

            <div>
                {messages.map((msg, i) => (
                    <div key={i}>
                        <strong>{msg.sender}:</strong> {msg.text}
                    </div>
                ))}
            </div>

            <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                        sendMessage(input);
                        setInput('');
                    }
                }}
            />
            <button onClick={() => {
                sendMessage(input);
                setInput('');
            }}>Send</button>
        </div>
    );
}
```

---

## CORS Configuration

The server has CORS enabled for all origins by default.

**For production, restrict to your domain:**

Edit `server.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Audio Formats

**Response Audio:**
- Format: WAV
- Sample Rate: 22050 Hz
- Channels: Mono
- Encoding: 16-bit PCM

**Input Audio (for `/api/voice-message`):**
- Supported: WAV, MP3, M4A, FLAC
- Recommended: WAV, 16kHz, mono

---

## Error Handling

**HTTP Errors:**
```javascript
try {
    const response = await fetch('http://YOUR_SERVER_IP:8001/api/text-message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: "Hello" })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
        console.error('API Error:', data.error);
    }
} catch (error) {
    console.error('Request failed:', error);
}
```

**WebSocket Errors:**
```javascript
ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
    if (!event.wasClean) {
        console.error('Connection died');
        // Implement reconnection logic
    }
};
```

---

## Testing

**Test text endpoint:**
```bash
curl -X POST http://YOUR_SERVER_IP:8001/api/text-message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, what'\''s your name?"}'
```

**Test WebSocket:**
```bash
# Install wscat: npm install -g wscat
wscat -c ws://YOUR_SERVER_IP:8001/ws
```

Then send:
```json
{"type": "text", "message": "Hello"}
```

---

## Rate Limiting (Recommended)

Add rate limiting in production:

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/text-message")
@limiter.limit("10/minute")  # 10 requests per minute
async def text_message(request: Request, data: dict):
    ...
```

---

## Security Recommendations

1. **Use HTTPS** in production
2. **Restrict CORS** to your domain
3. **Add API authentication** (API keys, JWT)
4. **Rate limit** all endpoints
5. **Validate input** (max message length)
6. **Monitor usage** (logging, analytics)
