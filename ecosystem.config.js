// PM2 Ecosystem Configuration for Flashback Avatar
// Usage: pm2 start ecosystem.config.js

module.exports = {
  apps: [
    {
      name: 'ollama-llm',
      script: 'ollama',
      args: 'serve',
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      env: {
        OLLAMA_HOST: '0.0.0.0:11434'
      }
    },
    {
      name: 'flashback-avatar',
      script: '/usr/bin/python3',
      args: 'flashback_avatar_server.py',
      cwd: '/mnt/FlashbackAvatars',
      autorestart: true,
      watch: false,
      max_memory_restart: '4G',
      interpreter: 'none',
      env: {
        PYTHONUNBUFFERED: '1'
      },
      error_file: './logs/avatar-error.log',
      out_file: './logs/avatar-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    }
  ]
};
