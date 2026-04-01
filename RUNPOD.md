# Running on RunPod

## First-time setup

### 1. Generate an SSH key on your local machine
```sh
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```
Copy the full output (starts with `ssh-ed25519 AAAA...`) and add it to RunPod → Settings → SSH Public Keys. **Deploy the pod after adding the key** — it only gets injected at boot time.

### 2. Spin up a pod
- Pods → Deploy → search for A100 (40GB is fine for small models)
- Use the **RunPod PyTorch** template
- Set disk storage to at least 20GB
- Deploy

### 3. SSH in
Get the SSH command from the pod's Connect page. It looks like:
```sh
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
```
When prompted "are you sure you want to continue connecting?" type `yes`.

### 4. Install rsync on the pod
```sh
apt-get update && apt-get install -y rsync && apt-get install -y tmux
```

### 5. Copy your repo from your laptop (run this locally, not on the pod) (also excludes TinyStories! If you want to exclude owt, then swap TinyStories for owt)
```sh
rsync -avz \
      --exclude='.git' \
      --exclude='.venv/' \
      --exclude='runs/' \
      --exclude='wandb/' \
      --exclude='cs336_basics/checkpoints/' \
      --exclude='cs336_basics/__pycache__/' \
      --exclude='*.pyc' \
      --exclude='data/TinyStories*' \
      --exclude='data/*.txt' \
      --exclude='cs336_basics/*.txt' \
      -e "ssh -p 26257 -i ~/.ssh/id_ed25519" \
      /Users/miguel/assignment1-basics/ \
      root@195.26.233.52:/workspace/assignment1-basics/
```

### 6. Set up the environment on the pod
```sh
cd /workspace/assignment1-basics
pip install uv
uv sync
```

### 7. Log in to W&B
```sh
export WANDB_API_KEY=your_key_here
# or
uv run wandb login --relogin
```
Note: W&B API keys are now longer than 40 characters. If you see a "key must be 40 characters" error, set `WANDB_API_KEY` as an env var instead.

---

## Running training


### Start a tmux session (so training survives SSH disconnects)
```sh
# Ghostty users need to override the terminal type
TERM=xterm-256color tmux new -s train
```

### Run training
```sh
uv run cs336_basics/main.py data/TinyStoriesV2-GPT4-train.npy \
  --val-dataset data/TinyStoriesV2-GPT4-valid.npy \
  --device cuda \
  --iterations 2500 \
  --val-every 250 \
  --save-every 500
```

### Detach from tmux (leave training running)
Press `Ctrl+B` then `D`

---

## Reconnecting after disconnect

```sh
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
TERM=xterm-256color tmux attach -t train
```

---

## Iteration count

Total tokens to process: 327,680,000 = `batch_size × iterations × context_length`

| batch_size | context_length | iterations |
|---|---|---|
| 64 | 256 | 20,000 |
| 512 | 256 | 2,500 |

---

## Syncing updated code to pod

```sh
rsync -avz \
  --exclude='.git' \
  --exclude='wandb/' \
  --exclude='cs336_basics/checkpoints/' \
  --exclude='cs336_basics/__pycache__/' \
  --exclude='*.pyc' \
  -e "ssh -p <port> -i ~/.ssh/id_ed25519" \
  /Users/miguel/assignment1-basics/ \
  root@<ip>:/workspace/assignment1-basics/
```
