import threading
import shutil
import faiss
import numpy as np
import json
import os
import logging
import time

logger = logging.getLogger("SpeakerDB")

class SpeakerProfileDB:
    def __init__(self, db_path="storage/speaker_db"):
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "speakers.index")
        self.metadata_file = os.path.join(db_path, "profiles.json")
        self.dim = 192
        self.lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize or Load
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.load()
        else:
            self.index = faiss.IndexFlatIP(self.dim) # Inner Product (Cosine Sim if normalized)
            self.profiles = {} # { "label": { "count": X, "last_seen": T } }
            self.id_map = []   # List where index i -> "Label"
            self.last_load_time = 0
            logger.info("Initialized New Speaker DB")
            
    def load(self):
        with self.lock:
            logger.info(f"Loading Speaker DB from {self.db_path}...")
            # We rely on JSON for Centroids. Index is derivative.
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.profiles = data.get("profiles", {})
                    # id_map is derived from rebuild
                    
                self._rebuild_index()
            else:
                logger.warning("No profile JSON found to load.")
                
            self.last_load_time = time.time()
            
    def save(self):
        with self.lock:
            # Atomic Save Strategy
            temp_index = self.index_file + ".tmp"
            temp_meta = self.metadata_file + ".tmp"
            
            # Write to Temp
            faiss.write_index(self.index, temp_index)
            with open(temp_meta, 'w') as f:
                json.dump({
                    "profiles": self.profiles,
                    "id_map": self.id_map
                }, f, indent=2)
                
            # Atomic Replace
            os.replace(temp_index, self.index_file)
            os.replace(temp_meta, self.metadata_file)
            
            # Update our load time so we don't reload our own save
            self.last_load_time = time.time()
            
    def identify(self, embedding, threshold=0.60):
        """
        Search for speaker.
        Returns: (label, score) or (None, score)
        """
        # Check for external updates (e.g. API renamed someone)
        # We check BEFORE lock to avoid contention on simple reads, 
        # but technically mtime check is racy. It's fine for this use case.
        if os.path.exists(self.metadata_file):
            mtime = os.path.getmtime(self.metadata_file)
            if mtime > self.last_load_time:
                logger.info("Detected DB update on disk. Reloading...")
                self.load() # This uses lock internally

        with self.lock:
            if self.index.ntotal == 0:
                return None, 0.0
                
            # FAISS expects [1, 192]
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
                
            # Search Top 1
            D, I = self.index.search(embedding, 1)
            score = float(D[0][0])
            idx = int(I[0][0])
            
            if score > threshold and idx >= 0 and idx < len(self.id_map):
                label = self.id_map[idx]
                return label, score
            
            return None, score
        
    def add_profile(self, label, embedding):
        """
        Add a new embedding vector for a label. 
        Uses Rolling Average Centroid strategy.
        1. Update Label Centroid.
        2. Rebuild FAISS Index from all Centroids.
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
            
        with self.lock:
            # 1. Update Centroid
            if label not in self.profiles:
                self.profiles[label] = { 
                    "count": 0, 
                    "last_seen": 0,
                    "vector": embedding.flatten().tolist() # Store as list for JSON serialization
                }
                new_vector = embedding.flatten()
                self.profiles[label]["count"] = 1
            else:
                # Running Average: New = (Old * N + New) / (N + 1)
                count = self.profiles[label]["count"]
                old_vector = np.array(self.profiles[label]["vector"], dtype=np.float32)
                
                new_vector = (old_vector * count + embedding.flatten()) / (count + 1)
                
                # Normalize (Cosine Similarity requires Unit Vectors)
                norm = np.linalg.norm(new_vector)
                if norm > 0:
                    new_vector = new_vector / norm
                    
                self.profiles[label]["vector"] = new_vector.tolist()
                self.profiles[label]["count"] += 1
            
            self.profiles[label]["last_seen"] = time.time()

            # 2. Rebuild Index (Small enough to do every time for <10k users)
            # We need to map Index IDs back to Labels. 
            # We'll regenerate self.id_map
            self._rebuild_index()
            
        self.save() 
        
    def _rebuild_index(self):
        """
        Internal: Rebuilds FAISS index from current profile centroids.
        Assumes Lock is held.
        """
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map = []
        
        # Collect vectors
        vectors = []
        labels = []
        
        # Sort profiles by label for consistent id_map order
        sorted_profiles = sorted(self.profiles.items())

        for label, data in sorted_profiles:
            if "vector" in data and data["vector"] is not None:
                vec = np.array(data["vector"], dtype=np.float32)
                if vec.shape[0] == self.dim:
                    vectors.append(vec)
                    labels.append(label)
        
        if vectors:
            matrix = np.stack(vectors)
            # Ensure normalized (should be already, but double check)
            faiss.normalize_L2(matrix)
            self.index.add(matrix)
            self.id_map = labels
        
        logger.info(f"Rebuilt Index with {self.index.ntotal} centroids.")

    def get_known_speakers(self):
        return list(self.profiles.keys())
