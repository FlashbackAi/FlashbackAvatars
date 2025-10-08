# Voice Chat Deployment Checklist

## ✅ Pre-Deployment Verification

### 1. Reference Audio Check
```bash
# Verify reference audio exists
ls -la ../avatar_input/vinay_audio.wav
```

**Status**: ✅ File exists at `avatar_input/vinay_audio.wav` (2.1 MB)

### 2. Install Dependencies
```bash
cd flashback_voice_chat
pip install -r requirements.txt
```

**Required packages**:
- `fastapi==0.104.1`
- `uvicorn[standard]==0.24.0`
- `websockets==12.0`
- `python-multipart==0.0.6`
- `requests==2.31.0`
- `TTS==0.21.3` (Coqui XTTS - **REQUIRED**)
- `openai-whisper==20231117` (for voice input)
- `pydub==0.25.1`
- `soundfile==0.12.1`

---

## 🚀 Start Server

### Option 1: Direct Start (Development)
```bash
cd flashback_voice_chat
python3 server.py
```

Server will start on: `http://0.0.0.0:8001`

### Option 2: PM2 (Production)
```bash
# Install PM2 (if not installed)
npm install -g pm2

# Start with PM2
cd /mnt/FlashbackAvatars/flashback_voice_chat
pm2 start ecosystem.voice.config.js

# Check status
pm2 status

# View logs
pm2 logs flashback-voice-chat

# Stop server
pm2 stop flashback-voice-chat

# Restart server
pm2 restart flashback-voice-chat
```

---

## 🧪 Testing

### 1. Test Server Health
```bash
curl http://localhost:8001/health
```

**Expected response**:
```json
{
  "status": "healthy",
  "service": "flashback-voice-chat"
}
```

### 2. Test Text Message API
```bash
curl -X POST http://localhost:8001/api/text-message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, what is your name?"}'
```

**Expected response**:
```json
{
  "type": "response",
  "text": "Good morning. I'm Vinay.",
  "audio_url": "/audio/audio_abc123.wav",
  "confidence": 1.0,
  "model": {...},
  "citations": [...]
}
```

### 3. Test WebSocket
```bash
# Install wscat if not installed
npm install -g wscat

# Connect to WebSocket
wscat -c ws://localhost:8001/ws
```

**Send test message**:
```json
{"type": "text", "message": "Hello"}
```

### 4. Test Web UI
Open browser: `http://localhost:8001`

**Test flow**:
1. Click "Connect" button
2. Type message in input box
3. Click "Send" or press Enter
4. Verify:
   - Message appears in chat
   - Waveform animation shows
   - Voice response plays

### 5. Test Voice Input (Optional)
```bash
curl -X POST http://localhost:8001/api/voice-message \
  -F "audio=@test_audio.wav"
```

---

## 🌐 External Website Integration

### Access from External IP

**From your website**, use:
- WebSocket: `ws://YOUR_SERVER_IP:8001/ws`
- REST API: `http://YOUR_SERVER_IP:8001/api/text-message`
- Voice API: `http://YOUR_SERVER_IP:8001/api/voice-message`

**CORS**: Already configured to allow all origins (`allow_origins=["*"]`)

For production, restrict to your domain in `server.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🔧 Troubleshooting

### Issue: "Reference audio not found"
**Solution**: Verify path in `server.py:153`:
```python
reference_audio = "../avatar_input/vinay_audio.wav"
```

### Issue: "TTS library not installed"
**Solution**:
```bash
pip install TTS==0.21.3
```

### Issue: "Whisper not found"
**Solution**:
```bash
pip install openai-whisper==20231117
```

### Issue: CUDA/GPU errors
**Solution**: XTTS will auto-fallback to CPU (slightly slower but works)

### Issue: Port 8001 already in use
**Solution**:
```bash
# Find process using port 8001
lsof -i :8001  # Linux/Mac
netstat -ano | findstr :8001  # Windows

# Kill process or change port in server.py:304
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Issue: WebSocket connection fails from browser
**Solution**: Check if using HTTPS. Update `index.html:329`:
```javascript
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
```

---

## 📊 Performance Notes

### XTTS Voice Generation
- **First request**: ~5-10 seconds (model loading)
- **Subsequent requests**: ~2-3 seconds
- **GPU**: ~1-2 seconds (if CUDA available)

### Whisper Transcription
- **Base model**: Fast, decent accuracy
- **Large model**: Slower, best accuracy (edit `server.py:222`)

### Memory Usage
- XTTS (CPU): ~1.5 GB RAM
- XTTS (GPU): ~2 GB VRAM
- Whisper (base): ~500 MB RAM

---

## 🔒 Security Checklist (Production)

- [ ] Change CORS to specific domain
- [ ] Add API authentication (JWT/API keys)
- [ ] Add rate limiting (slowapi)
- [ ] Use HTTPS/WSS (SSL certificates)
- [ ] Validate input (max message length)
- [ ] Monitor usage (logging)
- [ ] Set firewall rules (port 8001)

---

## 📝 Next Steps

1. **Deploy server**: Install dependencies and start
2. **Test locally**: Verify all endpoints work
3. **Test externally**: Access from your website
4. **Monitor**: Check PM2 logs for errors
5. **Optimize**: Fine-tune voice model if needed

---

## 🎯 Integration Complete

✅ Voice cloning with Vinay's voice (XTTS)
✅ WhatsApp-style call interface
✅ Real-time WebSocket communication
✅ REST API for external integration
✅ Speech-to-text capability (Whisper)
✅ API integration with RAG backend

**Server ready at**: `http://YOUR_SERVER_IP:8001`

See `EXTERNAL_INTEGRATION.md` for detailed integration examples.
