#!/usr/bin/env python
"""Quick API smoke test for AURA backend (manual script)."""

import asyncio
import io
import json
import os
import sys
from pathlib import Path

import httpx
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl_client.models.base_model import SimpleNet


async def _run_smoke_test() -> None:
    os.environ["AURA_MODE"] = "real"

    import main

    main.initialize_real_mode_components()
    model = SimpleNet()
    model_bytes_io = io.BytesIO()
    torch.save(model.state_dict(), model_bytes_io)
    model_payload = model_bytes_io.getvalue()

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        mode = await client.get("/system/mode")
        health = await client.get("/system/health")
        submit = await client.post(
            "/sentinel/submit_update",
            data={"hospital_id": "TEST_HOSP_001"},
            files={"model_file": ("model.pth", model_payload, "application/octet-stream")},
        )
        payload = submit.json()
        session = await client.get(f"/session/{payload['session_id']}")
        summary = await client.get("/stats/summary")

    report = {
        "mode_status": mode.status_code,
        "health_status": health.status_code,
        "submit_status": submit.status_code,
        "session_status": session.status_code,
        "summary_status": summary.status_code,
        "session_id": payload.get("session_id"),
        "verdict": payload.get("verdict"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(_run_smoke_test())
