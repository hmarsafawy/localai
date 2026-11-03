from anyio import to_thread
from fastapi import APIRouter, Query

from models.hardware import HardwareInfo
from schemas.hardware import HardwareResponse
from services.hardware import hardware_service

router = APIRouter()


@router.get("/hardware", response_model=HardwareResponse)
async def get_hardware(refresh: bool = Query(default=False)) -> HardwareInfo:
    """Everything relevant to running models on this machine. CPU, GPU list and
    disk types are cached (`refresh=true` re-reads them); free memory and free
    disk space are always current."""
    return await to_thread.run_sync(hardware_service.get_hardware, refresh)
