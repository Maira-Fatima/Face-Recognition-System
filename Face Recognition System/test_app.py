import urllib.request
import numpy as np
import cv2
import io
import os
from fastapi.testclient import TestClient
from app.main import app

def test_api():
    img_path = r"C:\Users\user\.gemini\antigravity-ide\brain\5b0bb8ea-d06c-43c2-8a3d-03b3399a0155\test_face_1782827707505.png"
    
    with TestClient(app) as client:
        # Test Health
        print("Testing /health...")
        response = client.get("/health")
        print(response.json())
        assert response.status_code == 200
        
        # Test Insert
        print("\nTesting /insert...")
        with open(img_path, "rb") as f:
            response = client.post(
                "/insert",
                data={"name": "Test Person"},
                files={"file": ("test_image.jpg", f, "image/jpeg")}
            )
        print(response.json())
        assert response.status_code == 200
        assert "record_id" in response.json()
        
        # Test Search
        print("\nTesting /search...")
        with open(img_path, "rb") as f:
            response = client.post(
                "/search",
                data={"top_k": 5},
                files={"file": ("test_image.jpg", f, "image/jpeg")}
            )
        print(response.json())
        assert response.status_code == 200
        assert "matches" in response.json()
        assert response.json()["best_match"] == "Test Person"

if __name__ == "__main__":
    test_api()
