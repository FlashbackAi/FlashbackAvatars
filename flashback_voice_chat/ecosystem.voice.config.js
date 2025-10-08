module.exports = {
  apps: [
    {
      name: 'flashback-voice-chat',
      script: 'python3',
      args: 'server.py',
      cwd: '/mnt/FlashbackAvatars/flashback_voice_chat',
      interpreter: 'none',
      env: {
        COQUI_TOS_AGREED: '1',
        PYTHONUNBUFFERED: '1'
      },
      error_file: '/root/.pm2/logs/voice-chat-error.log',
      out_file: '/root/.pm2/logs/voice-chat-out.log',
      max_memory_restart: '2G',
      restart_delay: 3000
    }
  ]
};
