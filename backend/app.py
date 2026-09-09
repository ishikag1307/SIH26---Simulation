"""FastAPI Backend Server for Telemetry, Video Streaming, and Scenario Controls."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.core.engine import ROOT, SimulationEngine


engine = SimulationEngine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.start()
    yield
    engine.stop()


app = FastAPI(title="SIH 2026 Virtual Camera Tracking Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    telemetry = engine.telemetry_snapshot()
    return {"status": "ok", "scenario": engine.scenario, "frame": telemetry.get("frame", 0)}


@app.post("/api/scenario")
def set_scenario(data: dict = Body(...)) -> dict:
    scenario = data.get("scenario", "uav-ground")
    engine.set_scenario(scenario)
    return {"status": "success", "scenario": scenario}


@app.post("/api/reset")
def reset_sim() -> dict:
    engine.reset()
    return {"status": "reset"}


@app.get("/api/export_log.csv")
def export_log() -> Response:
    csv_content = engine.tracker.generate_csv()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=fsoc_{engine.scenario}_log.csv"},
    )


def mjpeg_stream():
    while True:
        jpeg = engine.jpeg_snapshot()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / float(engine.config["fps"]))


@app.get("/api/camera.mjpeg")
def camera_stream() -> StreamingResponse:
    return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/telemetry")
async def telemetry_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            telemetry = engine.telemetry_snapshot()
            if telemetry:
                await websocket.send_json(telemetry)
            await asyncio.sleep(1.0 / 20.0)
    except WebSocketDisconnect:
        return


frontend_dist = Path(ROOT) / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
