# Flashback Voice Chat

WhatsApp-style voice interaction with Vinay's AI assistant.

## Features

✅ **Voice & Text Input**: Speak or type your questions
✅ **Voice Cloning**: Uses Vinay's voice (with reference audio)
✅ **Real-time Chat**: WebSocket-based instant responses
✅ **WhatsApp-style UI**: Familiar call interface
✅ **API Integration**: Connects to your RAG backend

## Quick Start

### 1. Install Dependencies

```bash
cd flashback_voice_chat
pip install -r requirements.txt
```

### 2. (Optional) Add Voice Cloning

If you have Vinay's voice sample:

```bash
# Put voice sample here:
cp /path/to/vinay_audio.wav ../avatar_input/vinay_audio.wav
```

Without it, system will use Microsoft edge-tts (still works great!)

### 3. Start Server

```bash
python server.py
```

Access at: http://localhost:8001

### 4. Deploy with PM2

```bash
# Create PM2 config
cat > ecosystem.voice.config.js << 'EOF'
module.exports = {
  apps: [{
    name: 'flashback-voice-chat',
    script: 'python',
    args: 'server.py',
    cwd: '/mnt/FlashbackAvatars/flashback_voice_chat',
    interpreter: 'none',
    env: {
      COQUI_TOS_AGREED: '1'
    }
  }]
};
EOF

# Start
pm2 start ecosystem.voice.config.js
pm2 save
```

## Usage

### Connect to Chat

1. Click "Connect" button
2. Wait for "Connected" status

### Text Input

1. Type message in input box
2. Press Enter or click "Send"

### Voice Input

1. Click 🎤 microphone button
2. Speak your question
3. System auto-sends when you finish speaking

### Mute/Unmute

- Click 🎤 to toggle mic
- 🔇 = muted
- 🎤 = active

### End Call

- Click red 📞 button to disconnect

## How It Works

```
User Input (Voice/Text)
    ↓
WebSocket → Server
    ↓
POST to RAG API (http://13.234.246.123:8188/answer/generate)
    ↓
Get Response
    ↓
Voice Synthesis (XTTS or edge-tts)
    ↓
Stream back to Client
    ↓
Play Audio + Show Text
```

## Customization

### Change API Endpoint

Edit `server.py`:

```python
voice_server = VoiceChatServer(
    api_url="YOUR_API_URL_HERE",
    reference_audio="path/to/audio.wav"
)
```

### Change Voice

**Option A: Voice Cloning (Best)**
- Provide 5-30 seconds of Vinay's voice
- System will clone it

**Option B: Edge-TTS Voices**

Edit `voice_cloning.py`:

```python
# Male voices
voice = "en-US-GuyNeural"
voice = "en-GB-RyanNeural"
voice = "en-IN-PrabhatNeural"  # Indian accent

# Female voices
voice = "en-US-JennyNeural"
voice = "en-IN-NeerjaNeural"  # Indian accent
```

## Troubleshooting

### XTTS CUDA Errors

If voice cloning fails with CUDA errors:

```bash
# Use CPU mode instead
export CUDA_VISIBLE_DEVICES=""
python server.py
```

Or system auto-falls back to edge-tts.

### WebSocket Connection Failed

Check firewall:

```bash
# Allow port 8001
sudo ufw allow 8001
```

### No Audio Playback

Check browser console for errors. Try different browser (Chrome recommended).

## Next Steps

Once this works, we can add:
1. Video avatar (integrate with avatar system later)
2. Better speech-to-text (Whisper integration)
3. Multi-language support
4. Conversation history
