# Real-Time Diarization

A comprehensive real-time speaker diarization system featuring a Python-based backend (integrating LiveKit for streaming audio) and a Next.js frontend for visualization and management.

## Project Structure

- **`realtime_diarization.py`**: A standalone script for local real-time diarization using your microphone and the Sortformer model. Useful for quick testing without the full server infrastructure.
- **`server/`**: The main backend logic.
    - **`bot.py`**: A LiveKit agent that connects to a room, processes incoming audio streams, and performs diarization.
    - **`api_server.py`**: A FastAPI backend that manages speaker profiles, sessions, and provides data to the frontend.
    - **`components/`**: Core logic including VAD, Overlap Detection, Embedding Extraction, etc.
- **`frontend/`**: A Next.js web application for verifying and managing diarization sessions.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Docker** (for MinIO and other infrastructure services)
- **LiveKit Server** (local or cloud)

## Setup & Installation

### 1. Environment Setup

Ensure you have the necessary environment variables. Check `.env` files in both `server/` and `frontend/` (copy `.env.example` if available).

### 2. Backend (Server)

```bash
cd server
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Frontend

```bash
cd frontend
npm install
```

## Running the Application

### Option A: Standalone Local Test
Run the local microphone test script (requires no external infrastructure):

```bash
python realtime_diarization.py
```

### Option B: Full System

#### 1. Start Infrastructure
Ensure MinIO and LiveKit server are running (usually via `docker-compose up` if provided, or manually).

#### 2. Run the API Server
This handles speaker management and session data.

```bash
# From the root directory or server directory
python server/api_server.py
```

#### 3. Run the LiveKit Agent (Bot)
This simply connects to the room and processes audio.

```bash
# From the root directory
python server/bot.py start
```

#### 4. Run the Frontend
Navigate to the frontend and start the development server.

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## Troubleshooting

- **Audio Issues**: Ensure `sounddevice` dependencies are installed (e.g., `libportaudio2` on Linux).
- **LiveKit Connection**: Verify `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` in `server/.env`.
- **MinIO**: Ensure the MinIO Docker container is running and buckets (`sessions`, `speakers`) are created.
