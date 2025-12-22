from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import os
from pathlib import Path
from datetime import datetime
import psycopg2
import psycopg2.extras

# --- Config ---
DB_CONFIG = {
    'host': 'localhost',
    'database': 'warning_data',
    'user': 'phidinh',
    'password': 'phi01478965',
    'port': 5432,
}

INFER_OUTPUT_DIR = Path("inference_output")

app = FastAPI(title="Fall Warning API", version="1.0.0")

# CORS cho phát triển (có thể hạn chế origin sau)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve ảnh tĩnh từ inference_output qua /images
INFER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(INFER_OUTPUT_DIR)), name="images")


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()} 


@app.get("/api/events")
def list_events(limit: int = 50):
    """Trả về danh sách fall_events mới nhất."""
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, event_time, event_type, confidence, image_path
            FROM fall_events
            ORDER BY event_time DESC
            LIMIT %s
            """,
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        # Bổ sung image_url để Flutter hiển thị trực tiếp
        results = []
        for r in rows:
            image_path = r.get("image_path")
            image_url = None
            if image_path:
                # Convert local path like "inference_output/xxx.jpg" to URL "/images/xxx.jpg"
                try:
                    p = Path(image_path)
                    image_url = f"/images/{p.name}"
                except Exception:
                    image_url = None
            results.append({
                "id": r.get("id"),
                "event_time": r.get("event_time").isoformat() if r.get("event_time") else None,
                "event_type": r.get("event_type"),
                "confidence": r.get("confidence"),
                "image_path": image_path,
                "image_url": image_url,
            })
        return {"items": results}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/api/events/{event_id}")
def get_event(event_id: int):
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, event_time, event_type, confidence, image_path
            FROM fall_events
            WHERE id = %s
            """,
            (event_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="Event not found")
        image_path = row.get("image_path")
        image_url = None
        if image_path:
            try:
                p = Path(image_path)
                image_url = f"/images/{p.name}"
            except Exception:
                image_url = None
        return {
            "id": row.get("id"),
            "event_time": row.get("event_time").isoformat() if row.get("event_time") else None,
            "event_type": row.get("event_type"),
            "confidence": row.get("confidence"),
            "image_path": image_path,
            "image_url": image_url,
        }
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
