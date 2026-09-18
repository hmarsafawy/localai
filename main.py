from fastapi import FastAPI

from routes.model_router import router as model_router


app = FastAPI(
    title="LocalAI Model Catalog",
    description="Search and inspect models available on Hugging Face.",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(model_router, prefix="/models", tags=["Models"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
