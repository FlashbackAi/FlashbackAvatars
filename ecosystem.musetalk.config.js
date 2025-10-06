module.exports = {
  apps: [
    {
      name: 'ollama-llm',
      script: 'ollama',
      args: 'serve',
      autorestart: true,
      env: {
        OLLAMA_HOST: '0.0.0.0:11434'
      }
    },
    {
      name: 'flashback-musetalk',
      script: 'python3',
      args: 'flashback_production_musetalk.py',
      cwd: process.env.PWD || '/mnt/FlashbackAvatars',
      autorestart: true,
      max_memory_restart: '8G',
      env: {
        PYTHONUNBUFFERED: '1',
        COQUI_TOS_AGREED: '1'  // Accept Coqui XTTS license (free for non-commercial use)
      }
    }
  ]
};
