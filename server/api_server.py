from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import os
import json
import numpy as np
import io

from speaker_db import SpeakerProfileDB
from storage.minio_client import MinioUploader

# Initialize
app = FastAPI(title="LiveDiar API")
logger = logging.getLogger("API")
logging.basicConfig(level=logging.INFO)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Resources
speaker_db = SpeakerProfileDB()
uploader = MinioUploader()

class AssignRequest(BaseModel):
    label: str # Real Name
    vector_path: str # Path to JSON vector
    delete_original: bool = False

@app.get("/speakers")
def get_speakers():
    """List all known speakers and their metadata."""
    profiles = speaker_db.profiles
    result = []
    # Force auto-reload check
    if os.path.exists(speaker_db.metadata_file):
        if os.path.getmtime(speaker_db.metadata_file) > speaker_db.last_load_time:
             speaker_db.load()

    for label, data in profiles.items():
        result.append({
            "label": label,
            "count": data.get("count", 0),
            "last_seen": data.get("last_seen", 0)
        })
    # Sort by last_seen
    result.sort(key=lambda x: x["last_seen"], reverse=True)
    return result

@app.get("/sessions/active")
def get_active_session():
    """
    Get the most recent session from MinIO (Assuming it's the active one).
    """
    if not uploader.client:
        raise HTTPException(status_code=503, detail="MinIO failed")
    
    # List folders in 'sessions/'
    # MinIO list_objects returned objects. We need to parse prefixes.
    # Actually, we can just look for metadata.json files and sort by date.
    # This is slow if many sessions.
    # Better: Identify based on 'most recent modification'.
    try:
        # Recursive list? No, limit.
        objects = uploader.client.list_objects(uploader.bucket, prefix="sessions/", recursive=True)
        # Filter for metadata.json
        sessions = []
        for obj in objects:
            if obj.object_name.endswith("metadata.json"):
                # sessions/SESSION_ID/metadata.json
                parts = obj.object_name.split("/")
                if len(parts) >= 3:
                     sid = parts[1]
                     sessions.append({
                         "id": sid,
                         "last_modified": obj.last_modified
                     })
        
        if not sessions:
            return {"active_session": None}
            
        sessions.sort(key=lambda x: x["last_modified"], reverse=True)
        return {"active_session": sessions[0]["id"]}
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/candidates")
def get_candidates(session_id: str):
    """
    List candidate speakers (audio snippets) for a session.
    """
    if not uploader.client:
        raise HTTPException(status_code=503, detail="MinIO failed")

    prefix = f"sessions/{session_id}/segments/"
    candidates = []
    
    try:
        objects = uploader.client.list_objects(uploader.bucket, prefix=prefix)
        for obj in objects:
            if obj.object_name.endswith(".wav"):
                # Object: sessions/{id}/segments/spk_0.wav
                filename = os.path.basename(obj.object_name)
                spk_id = filename.replace(".wav", "")
                
                # Check for vector existence (spk_0.json)
                vec_path = obj.object_name.replace(".wav", ".json")
                # We can't easily check check existence without a head call?
                # Just assume it exists if wav exists (bot uploads both).
                
                # Generate Presigned URL
                url = uploader.get_presigned_url(obj.object_name)
                
                candidates.append({
                    "spk_id": spk_id,
                    "audio_url": url,
                    "vector_path": vec_path
                })
                
        return candidates
    except Exception as e:
         logger.error(f"Failed to list candidates: {e}")
         return []

@app.post("/speakers/assign")
def assign_speaker(req: AssignRequest):
    """
    Assign a real name to a candidate.
    1. Downloads vector from MinIO.
    2. Updates Profile DB (Centroid).
    """
    try:
        # 1. Download Vector
        # We need to get the object bytes.
        response = uploader.client.get_object(uploader.bucket, req.vector_path)
        data = json.loads(response.read())
        vector = np.array(data["vector"], dtype=np.float32)
        response.close()
        
        # 2. Add to DB
        speaker_db.add_profile(req.label, vector)
        
        return {"status": "success", "message": f"Assigned {req.label}"}
        
    except Exception as e:
        logger.error(f"Assignment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
