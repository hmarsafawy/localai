from fastapi import APIRouter, HTTPException, Query
from anyio import to_thread
from huggingface_hub.utils import HfHubHTTPError

from models.hf_model import HfModel, HfModelRequirements, HfPipelineInfo
from schemas.hf_model import (
    HfModelResponse,
    HfModelSearchRequest,
    HfModelSearchResponse,
    HfPipelineResponse,
    HfRequirementsResponse,
)
from services.hardware import hardware_service
from services.huggingface import HuggingFaceService, ModelIndexNotFoundError
from services.model_options import annotate_fit

router = APIRouter()
service = HuggingFaceService()


def _requirements_with_fit(
    model_id: str, resolve_base_model: bool, fingerprint: bool, fit: bool
) -> HfModelRequirements:
    requirements = service.get_model_requirements(model_id, resolve_base_model, fingerprint)
    annotate_fit(requirements, hardware_service.memory_budget() if fit else None)
    return requirements


def _map_hf_error(error: HfHubHTTPError) -> HTTPException:
    """`HfHubHTTPError` covers everything from a real 404 to a gated repo to a
    rate limit - collapsing all of that into a blanket "not found" is
    misleading, so surface what the Hub actually said instead."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status == 404:
        return HTTPException(status_code=404, detail="Hugging Face model not found")
    if status in (401, 403):
        return HTTPException(
            status_code=403, detail="This repository is gated or private and could not be accessed"
        )
    if status == 429:
        return HTTPException(status_code=429, detail="Rate limited by the Hugging Face Hub, try again shortly")
    return HTTPException(status_code=502, detail=f"Hugging Face Hub error ({status or 'unknown'}): {error}")


@router.get("/", response_model=HfModelSearchResponse)
async def search_models(filters: HfModelSearchRequest = Query()) -> HfModelSearchResponse:
    items = await to_thread.run_sync(service.search_models, filters)
    return HfModelSearchResponse(
        items=items[: filters.limit],
        limit=filters.limit,
        has_more=len(items) > filters.limit,
    )


@router.get("/{model_id:path}/requirements", response_model=HfRequirementsResponse)
async def get_model_requirements(
    model_id: str,
    resolve_base_model: bool = Query(default=True),
    fingerprint: bool = Query(default=True),
    fit: bool = Query(default=True),
) -> HfModelRequirements:
    """What a model needs to run, even without a `model_index.json`: falls back
    through the repo's `base_model` ancestry, weight tensor-name fingerprinting,
    and a small curated table of known pipeline families (Flux, SDXL, ...).

    Each role's `options` lists what can be downloaded for it, one per
    quantization, precision or variant. With `fit`, LLM options are also rated
    against this machine's free memory and one is marked as recommended."""
    try:
        return await to_thread.run_sync(
            _requirements_with_fit, model_id, resolve_base_model, fingerprint, fit
        )
    except HfHubHTTPError as error:
        raise _map_hf_error(error) from error


@router.get("/{model_id:path}/pipeline", response_model=HfPipelineResponse)
async def get_model_pipeline(model_id: str, include_config: bool = Query(default=True)) -> HfPipelineInfo:
    """Parse the repo's `model_index.json` (or `modular_model_index.json`) and report what the pipeline needs
    (text encoder, VAE, scheduler, ...) plus whether each piece is actually in the repo."""
    try:
        return await to_thread.run_sync(service.get_pipeline_requirements, model_id, include_config)
    except ModelIndexNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="Repository has no model_index.json or modular_model_index.json (not a diffusers pipeline)",
        ) from error
    except HfHubHTTPError as error:
        raise _map_hf_error(error) from error


@router.get("/{model_id:path}", response_model=HfModelResponse)
async def get_model(model_id: str) -> HfModel:
    try:
        return await to_thread.run_sync(service.get_model, model_id)
    except HfHubHTTPError as error:
        raise _map_hf_error(error) from error
