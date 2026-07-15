// PM2 deployment for the v4 challenger miner (runs ALONGSIDE v3, different UID/port).
// Usage:  pm2 start serve/ecosystem.config.js   (from 04_our_miner_v4/)
// Config lives in 04_our_miner_v4/.env (loaded by python-dotenv in-process).
// Shares the v3 venv via the .venv symlink; code + artifacts are fully separate.

const ROOT = "/root/Skip/poker/SN126/04_our_miner_v4";
const PY = ROOT + "/.venv/bin/python";

module.exports = {
  apps: [
    {
      name: "p44_miner_v4",
      cwd: ROOT,
      script: PY,
      interpreter: "none",
      args: ["-m", "serve.miner", "--logging.info"],
      autorestart: true,
      max_restarts: 50,
      restart_delay: 15000,
      env: { PYTHONPATH: ROOT },
    },
    {
      name: "p44_retrain_v4",
      cwd: ROOT,
      script: PY,
      interpreter: "none",
      args: ["-m", "pipeline.retrain", "--daemon"],
      autorestart: true,
      max_restarts: 20,
      restart_delay: 60000,
      env: { PYTHONPATH: ROOT },
    },
  ],
};
