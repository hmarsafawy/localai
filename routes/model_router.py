from fastapi import APIRouter, HTTPException, Query
from anyio import to_thread
from huggingface_hub.utils import HfHubHTTPError

from models.hf_model import HfModel, HfPipelineInfo
from schemas.hf_model import (
    HfModelResponse,
    HfModelSearchRequest,
    HfModelSearchResponse,
    HfPipelineResponse,
)
from services.huggingface import HuggingFaceService, ModelIndexNotFoundError

router = APIRouter()
service = HuggingFaceService()


@router.get("/", response_model=HfModelSearchResponse)
async def search_models(filters: HfModelSearchRequest = Query()) -> HfModelSearchResponse:
    items = await to_thread.run_sync(service.search_models, filters)
    return HfModelSearchResponse(
        items=items[: filters.limit],
        limit=filters.limit,
        has_more=len(items) > filters.limit,
    )


@router.get("/{model_id:path}/pipeline", response_model=HfPipelineResponse)
async def get_model_pipeline(model_id: str, include_config: bool = Query(default=True)) -> HfPipelineInfo:
    """Parse the repo's `model_index.json` and report what the pipeline needs
    (text encoder, VAE, scheduler, ...) plus whether each piece is actually in the repo."""
    try:
        return await to_thread.run_sync(service.get_pipeline_requirements, model_id, include_config)
    except ModelIndexNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="Repository does not have a model_index.json (not a diffusers pipeline)",
        ) from error
    except HfHubHTTPError as error:
        raise HTTPException(status_code=404, detail="Hugging Face model not found") from error


@router.get("/{model_id:path}", response_model=HfModelResponse)
async def get_model(model_id: str) -> HfModel:
    try:
        return await to_thread.run_sync(service.get_model, model_id)
    except HfHubHTTPError as error:
        raise HTTPException(status_code=404, detail="Hugging Face model not found") from error
