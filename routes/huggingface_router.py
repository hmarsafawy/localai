from anyio import to_thread
from fastapi import APIRouter, HTTPException, Query
from huggingface_hub.utils import HfHubHTTPError

from models.hf_model import HfModel, HfModelRequirements
from schemas.hf_model import (
    HfModelResponse,
    HfModelSearchRequest,
    HfModelSearchResponse,
    HfRequirementsResponse,
)
from services.huggingface import HuggingFaceService
from services.llama_runtime import llama_runtime_service
from services.model_options import annotate_fit

router = APIRouter()
service = HuggingFaceService()


def _requirements_with_fit(
    model_id: str,
    resolve_base_model: bool,
    fingerprint: bool,
    fit: bool,
    include_config: bool,
    context_size: int | None,
) -> HfModelRequirements:
    requirements = service.get_model_requirements(
        model_id, resolve_base_model, fingerprint, include_config=include_config
    )
    annotate_fit(requirements, llama_runtime_service.memory_budget() if fit else None, context_size)
    return requirements


def map_hf_error(error: HfHubHTTPError) -> HTTPException:
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
    include_config: bool = Query(default=False),
    context_size: int | None = Query(default=None, ge=512, le=4_194_304),
) -> HfModelRequirements:
    """What a model needs to run, and what each part of it is.

    A diffusers pipeline is read from its `model_index.json` (or
    `modular_model_index.json`). That gives each role's `architecture` and
    `library`, plus the pipeline's `diffusers_version` and its other settings
    in `pipeline_config`. Without an index, this falls back through the repo's
    `base_model` ancestry, weight tensor-name fingerprinting, and a small
    curated table of known pipeline families (Flux, SDXL, ...).

    Each role's `options` lists what can be downloaded for it, one per
    quantization, precision or variant. An LLM option's `memory` says what it
    takes to run, term by term, read from the model's own layers: for
    `context_size` tokens, or 16,384 by default, but never more than the
    model was trained for. With `fit`, LLM options are also rated against this
    machine's free memory, `max_gpu_context` gives the longest context that keeps
    each on the GPU, and one is marked as recommended. With `include_config`,
    each pipeline component's own config is added as its role's `config`. That
    downloads one file per component, so it's off by default."""
    try:
        return await to_thread.run_sync(
            _requirements_with_fit, model_id, resolve_base_model, fingerprint, fit, include_config, context_size
        )
    except HfHubHTTPError as error:
        raise map_hf_error(error) from error


@router.get("/{model_id:path}", response_model=HfModelResponse)
async def get_model(model_id: str) -> HfModel:
    try:
        return await to_thread.run_sync(service.get_model, model_id)
    except HfHubHTTPError as error:
        raise map_hf_error(error) from error
