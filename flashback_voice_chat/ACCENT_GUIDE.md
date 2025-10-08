# Indian English Accent Guide

## How XTTS Handles Accents

XTTS **automatically clones the accent** from your reference audio (`vinay_audio.wav`). There's no separate "en_IN" language code - the accent comes from the reference audio itself.

## ✅ Current Setup

Your voice cloning is configured to preserve the Indian English accent from `vinay_audio.wav`.

### Settings for Best Accent Preservation

Edit `flashback_voice_chat/config.py`:

```python
VOICE_SETTINGS = {
    "temperature": 0.65,        # Lower = better accent preservation
    "repetition_penalty": 10.0,  # Natural Indian English speech patterns
    # ... other settings
}
```

---

## 🎯 Improving Indian English Accent Quality

### 1. Check Reference Audio Quality

Your `vinay_audio.wav` should have:
- ✅ **Clear Indian English accent**
- ✅ **5-30 seconds duration** (longer = better)
- ✅ **No background noise**
- ✅ **Natural speech** (not reading monotone)
- ✅ **16kHz+ sample rate**

**Check your audio:**
```bash
# Check audio properties
ffmpeg -i ../avatar_input/vinay_audio.wav
```

**Expected output:**
```
Duration: 00:00:10-00:00:30 (good)
Sample rate: 16000 Hz or higher (good)
Channels: mono (good)
```

---

### 2. Adjust Temperature for Accent

Lower temperature = better accent matching.

Edit `flashback_voice_chat/config.py`:

```python
VOICE_SETTINGS = {
    # For stronger Indian accent preservation:
    "temperature": 0.60,  # Try 0.60 (was 0.65)

    # If accent sounds too robotic, increase slightly:
    "temperature": 0.70,  # More natural but less accurate

    # Current (balanced):
    "temperature": 0.65,  # Good balance
}
```

**Test different values:**
- 0.50-0.60: Most accurate accent, may sound slightly robotic
- 0.65-0.70: Balanced (recommended)
- 0.75-0.85: Natural but accent may drift

---

### 3. Improve Reference Audio

If the cloned voice doesn't sound like Vinay's Indian accent:

**Option A: Re-record reference audio**

Record 15-30 seconds of Vinay speaking naturally in Indian English:
- Clear environment (no background noise)
- Natural conversational tone
- Include variety of sounds (vowels, consonants)
- Don't read from script - speak naturally

**Save as:**
```
avatar_input/vinay_audio.wav
```

**Convert if needed:**
```bash
# Convert to proper format
ffmpeg -i input.mp3 -ar 16000 -ac 1 avatar_input/vinay_audio.wav
```

**Option B: Extract better clip from existing video**

If you have video of Vinay speaking:
```bash
# Extract 15 seconds of audio from video
ffmpeg -i vinay_video.mp4 -ss 00:00:05 -t 00:00:15 -ar 16000 -ac 1 vinay_audio.wav
```

---

### 4. Test Accent Quality

**Quick test:**
```bash
cd flashback_voice_chat
python -c "
import asyncio
from voice_cloning import VoiceCloner

async def test():
    cloner = VoiceCloner(reference_audio='../avatar_input/vinay_audio.wav')
    await cloner.synthesize(
        'Hello, this is Vinay from Flashback Labs.',
        'test_accent.wav'
    )
    print('Generated: test_accent.wav')

asyncio.run(test())
"
```

**Listen to:** `test_accent.wav`

**Does it sound like Vinay's Indian accent?**
- ✅ Yes → Settings are good!
- ❌ No → Try lower temperature (0.60) or improve reference audio

---

## 🔧 Fine-Tuning Settings

### For Stronger Indian Accent

Edit `config.py`:
```python
VOICE_SETTINGS = {
    "temperature": 0.60,          # ← Lower for stronger accent
    "repetition_penalty": 12.0,    # ← Higher for clearer speech
    "top_p": 0.80,                 # ← Lower for more consistency
    "speed": 0.95,                 # ← Slightly slower (optional)
}
```

### For More Natural Sound (Slightly Less Accurate)

Edit `config.py`:
```python
VOICE_SETTINGS = {
    "temperature": 0.70,           # ← Higher for more variation
    "repetition_penalty": 8.0,     # ← Lower for more natural flow
    "top_p": 0.90,                 # ← Higher for more diversity
    "speed": 1.0,                  # ← Normal speed
}
```

---

## 📊 Accent Quality Comparison

| Temperature | Accent Accuracy | Naturalness | Recommended For |
|-------------|----------------|-------------|-----------------|
| 0.50-0.60   | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good | Exact accent match needed |
| **0.65** ✨ | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐⭐ Very Good | **Recommended (balanced)** |
| 0.70-0.75   | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | More expressive speech |
| 0.80-0.90   | ⭐⭐ Fair | ⭐⭐⭐⭐⭐ Excellent | Natural variation, less accurate |

**Current setting: 0.65** (balanced for Indian English accent)

---

## 🎤 Recording Tips for Best Indian Accent

If you're re-recording reference audio:

1. **Speak naturally** in Indian English
2. **Include typical phrases**: "I will do the needful", "What is your good name?", etc.
3. **Show intonation patterns** specific to Indian English
4. **Clear pronunciation** of both Indian and Western words
5. **No code-switching** - stay in Indian English throughout
6. **15-30 seconds** of natural speech

**Example script for recording:**
```
"Hello, I'm Vinay Thadem, co-founder of Flashback Labs.
We're building innovative solutions for avatar technology.
How can I help you today? Feel free to ask me anything
about our products and services."
```

---

## 🧪 Testing Workflow

1. **Record/improve reference audio**
   ```bash
   # Save as avatar_input/vinay_audio.wav
   ```

2. **Adjust temperature** in `config.py`
   ```python
   "temperature": 0.60  # Start here
   ```

3. **Test generation**
   ```bash
   python server.py
   # Test in browser
   ```

4. **Listen and adjust**:
   - Accent not strong enough? → Lower temperature (0.55-0.60)
   - Sounds robotic? → Increase temperature (0.70-0.75)
   - Accent drifts? → Check reference audio quality

5. **Repeat until satisfied**

---

## ⚠️ Important Notes

1. **XTTS has NO "en_IN" code** - accent comes from reference audio
2. **Quality of reference audio = Quality of cloned accent**
3. **Temperature is the main control** for accent preservation
4. **Lower temperature** = better accent match but less natural variation
5. **Reference audio should be 5-30 seconds** for best results

---

## 🚀 Quick Fix Checklist

If accent doesn't sound right:

- [ ] Reference audio has clear Indian accent?
- [ ] Reference audio is 10-30 seconds long?
- [ ] Reference audio has no background noise?
- [ ] Temperature set to 0.60-0.65?
- [ ] Repetition penalty set to 10+?
- [ ] Tested with multiple sentences?
- [ ] Compared to Coqui.ai web app with same audio?

If all checked and still not matching:
→ **Record new reference audio** with clearer Indian accent

---

## 📝 Summary

**Current Configuration:**
- Temperature: **0.65** (balanced)
- Repetition Penalty: **10.0** (natural)
- Language: **en** (accent from reference audio)

**To strengthen Indian accent:**
```python
# Edit config.py
"temperature": 0.60  # Lower temperature
```

**Remember:** The accent quality depends mainly on:
1. Reference audio quality (most important!)
2. Temperature setting (0.60-0.65 recommended)
3. Speaking clearly in generated text

Happy cloning! 🎙️
