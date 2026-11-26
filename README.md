# GoalScope

Football video analysis with player and ball tracking, a two-stage player selection workflow, tactical visualization, and multimodal highlight generation.

## Run locally

1. Create a Python 3.10 environment and install PyTorch and torchvision for your platform.
2. Install the backend dependencies with `python -m pip install -r backend/requirements.txt`.
3. Place model weights from your own or authorized sources in the project. The tracking pipeline looks for a player detector at `backend/football_player.pt` or `yolov8n.pt` in the root. Place `football-ball-detection.pt` in the root for the dedicated ball detector. Optional OSNet weights can also be placed in the root.
4. Run `START.bat` on Windows or run `python main.py` from `backend`.
5. Open `http://127.0.0.1:9999`. Register your own account or use guest mode.

SQLite, video uploads, and analysis caches are created under `GoalScope_data/` by default. The application is intended for local use.
