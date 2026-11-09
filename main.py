from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from anyio import to_thread
from fastapi import FastAPI

from routes.download_router import router as download_router
from routes.library_router import router as library_router
from routes.model_router import router as model_router
from routes.runtime_router import router as runtime_router
from routes.server_router import router as server_router
from routes.system_router import router as system_router
from services.hardware import hardware_service
from services.llama_servers import llama_server_service


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    hardware_service.preload()
    yield
    # A model left running would hold on to GPU memory after the app has gone.
    await to_thread.run_sync(llama_server_service.stop_all)


app = FastAPI(
    title="LocalAI",
    description=(
        "Find models on Hugging Face, see what this machine can run, download them, "
        "manage the runtimes that run them, and run them."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(model_router, prefix="/models", tags=["Models"])
app.include_router(download_router, prefix="/downloads", tags=["Downloads"])
app.include_router(library_router, prefix="/library", tags=["Library"])
app.include_router(system_router, prefix="/system", tags=["System"])
app.include_router(runtime_router, prefix="/runtimes/llama.cpp", tags=["Runtimes"])
app.include_router(server_router, prefix="/servers", tags=["Servers"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
