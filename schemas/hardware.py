from __future__ import annotations

from pydantic import ConfigDict

from models.hardware import HardwareInfo


class HardwareResponse(HardwareInfo):
    """Public response shape for the machine's hardware snapshot."""

    model_config = ConfigDict(from_attributes=True)
