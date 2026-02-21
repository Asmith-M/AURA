from typing import Any, Dict


async def demo_submission_response(hospital_id: str) -> Dict[str, Any]:
    raise RuntimeError("Demo mode has been removed; real pipeline is required.")


get_demo_submission_response = demo_submission_response
