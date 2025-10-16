"""
Voice Chat Configuration
Adjust these settings to fine-tune voice quality and accent
"""

# Voice Cloning Settings
# These settings match Coqui web app (https://coquitts.com/generate) for best accent cloning
VOICE_SETTINGS = {
    # Temperature: Lower = more consistent/accurate accent (0.1-1.0)
    # 0.1-0.3: Web app setting - BEST for accent preservation (strict cloning)
    # 0.65: More natural but loses accent
    # Try 0.1 for strongest accent cloning (may sound less natural)
    "temperature": 0.1,  # Changed from 0.65 to match web app

    # Repetition Penalty: Higher = less repetitive (1.0-20.0)
    # 10.0: Coqui.ai default, good for natural speech
    "repetition_penalty": 10.0,

    # Top K: Sampling diversity (10-100)
    # 50: Good balance
    "top_k": 50,

    # Top P: Nucleus sampling (0.1-1.0)
    # 0.85: Recommended
    "top_p": 0.85,

    # Speed: Speech rate multiplier (0.5-2.0)
    # 1.0: Normal speed
    "speed": 1.5,

    # Length Penalty: Controls output length
    # 1.0: Default
    "length_penalty": 1.0,

    # Enable text splitting for long sentences
    "enable_text_splitting": True,

    # Enable sampling (required for natural speech)
    "do_sample": True,
}

# For Indian English accent:
# - Ensure reference audio (vinay_audio.wav) has clear Indian accent
# - Lower temperature (0.60-0.65) preserves accent better
# - Higher repetition_penalty (10+) makes speech more natural

# Language code
LANGUAGE = "en"  # English (accent from reference audio)

# Reference audio requirements for best accent cloning:
# - Duration: 5-30 seconds
# - Quality: Clear, no background noise
# - Content: Natural speech with accent features
# - Sample rate: 16kHz or higher
