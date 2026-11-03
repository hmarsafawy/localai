from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from routes.model_router import router as model_router
from routes.runtime_router import router as runtime_router
from routes.system_router import router as system_router
from services.hardware import hardware_service


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    hardware_service.preload()
    yield


app = FastAPI(
    title="LocalAI",
    description=(
        "Find models on Hugging Face, see what this machine can run, and manage the runtimes that run them."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(model_router, prefix="/models", tags=["Models"])
app.include_router(system_router, prefix="/system", tags=["System"])
app.include_router(runtime_router, prefix="/runtimes/llama.cpp", tags=["Runtimes"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
