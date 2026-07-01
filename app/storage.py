import os
import sqlite3
import datetime
import faiss
import numpy as np
from app.config import settings

class FaceStore:
    """
    Manages the FAISS index for vector similarity search and a SQLite database
    for storing associated metadata.
    """
    def __init__(self, engine_name: str = settings.DEFAULT_ENGINE):
        self.engine_name = engine_name
        self.db_path = settings.DB_PATH
        # We can maintain separate FAISS indices per engine for clean benchmarking
        self.faiss_index_path = os.path.join(settings.DATA_DIR, f"{engine_name}_faiss.bin")
        
        self.dim = settings.EMBEDDING_DIM
        
        self._init_sqlite()
        self._init_faiss()

    def _init_sqlite(self):
        """Initialize the SQLite database and create the table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    image_path TEXT,
                    engine TEXT NOT NULL,
                    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def _init_faiss(self):
        """Initialize the FAISS index, loading from disk if available."""
        if os.path.exists(self.faiss_index_path):
            self.index = faiss.read_index(self.faiss_index_path)
        else:
            # IndexFlatIP uses Inner Product (Cosine similarity if vectors are L2-normalized)
            base_index = faiss.IndexFlatIP(self.dim)
            # IndexIDMap allows us to assign custom integer IDs mapping to SQLite IDs
            self.index = faiss.IndexIDMap(base_index)
            
    def _save_faiss(self):
        """Persist the FAISS index to disk."""
        faiss.write_index(self.index, self.faiss_index_path)

    def insert(self, name: str, embedding: np.ndarray, image_path: str = None) -> int:
        """
        Inserts a face embedding and metadata into the store.
        
        Args:
            name (str): The label/name of the person.
            embedding (np.ndarray): The embedding vector (must be shape (1, dim)).
            image_path (str, optional): Original image path.
            
        Returns:
            int: The inserted record ID.
        """
        if embedding.shape != (1, self.dim):
            embedding = embedding.reshape(1, self.dim)
            
        # Ensure it is float32 for FAISS
        embedding = embedding.astype(np.float32)

        # 1. Insert metadata into SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO faces (name, image_path, engine) VALUES (?, ?, ?)",
                (name, image_path, self.engine_name)
            )
            record_id = cursor.lastrowid
            conn.commit()

        # 2. Insert vector into FAISS
        ids = np.array([record_id], dtype=np.int64)
        self.index.add_with_ids(embedding, ids)
        
        # 3. Persist FAISS index
        self._save_faiss()
        
        return record_id

    def search(self, embedding: np.ndarray, top_k: int = 5):
        """
        Searches for the top-k most similar faces.
        
        Args:
            embedding (np.ndarray): The query embedding (must be shape (1, dim)).
            top_k (int): Number of results to return.
            
        Returns:
            list of dicts containing similarity scores and metadata.
        """
        if self.index.ntotal == 0:
            return []
            
        if embedding.shape != (1, self.dim):
            embedding = embedding.reshape(1, self.dim)
            
        embedding = embedding.astype(np.float32)
        
        # Search FAISS
        # D is distances (inner products), I is indices (mapped IDs)
        distances, ids = self.index.search(embedding, top_k)
        
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for dist, record_id in zip(distances[0], ids[0]):
                if record_id == -1:  # FAISS returns -1 if not enough results found
                    continue
                    
                cursor.execute("SELECT id, name, image_path FROM faces WHERE id = ?", (int(record_id),))
                row = cursor.fetchone()
                if row:
                    results.append({
                        "id": row[0],
                        "name": row[1],
                        "image_path": row[2],
                        "similarity": float(dist)
                    })
                    
        return results

    def get_total_faces(self):
        return self.index.ntotal
