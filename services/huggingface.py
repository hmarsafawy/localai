from __future__ import annotations

import json
import os
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from models.hf_model import HfModel, HfModelArtifact, HfPipelineComponent, HfPipelineFile, HfPipelineInfo
from schemas.hf_model import HfModelSearchRequest

MODEL_INDEX_FILENAME = "model_index.json"

WEIGHT_FORMATS_BY_EXTENSION = {
    ".safetensors": "safetensors",
    ".bin": "pytorch",
    ".pt": "pytorch",
    ".ckpt": "pytorch",
    ".msgpack": "flax",
    ".onnx": "onnx",
    ".onnx_data": "onnx",
    ".xml": "openvino",
}

COMPONENT_CONFIG_FILENAMES = (
    "config.json",
    "scheduler_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)


class ModelIndexNotFoundError(Exception):
    """Raised when a repo has no `model_index.json` (not a diffusers pipeline)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"{model_id} does not provide a model_index.json")


class HuggingFaceService:
    def __init__(self, api: HfApi | None = None) -> None:
        self.api = api or HfApi()

    def search_models(self, request: HfModelSearchRequest) -> list[HfModel]:
        filters = list(request.tags)
        if request.task:
            filters.append(request.task)
        if request.library:
            filters.append(request.library)
        if request.languages:
            filters.extend(request.languages)

        models = list(self.api.list_models(
            search=request.search,
            author=request.author,
            filter=filters or None,
            gated=request.gated,
            sort=request.sort,
            limit=request.limit + 1,
            expand=[
                "author",
                "createdAt",
                "downloads",
                "gated",
                "lastModified",
                "library_name",
                "likes",
                "pipeline_tag",
                "private",
                "sha",
                "tags",
                "trendingScore",
            ],
        ))
        if request.direction == 1:
            models.reverse()
        return [self._summary_from_info(model) for model in models]

    def get_model(self, model_id: str) -> HfModel:
        info = self.api.model_info(model_id, files_metadata=True)
        summary = self._summary_from_info(info)
        config = self._metadata_dict(getattr(info, "config", None))
        card_data = self._metadata_dict(getattr(info, "cardData", None))
        siblings = [
            HfModelArtifact.model_validate(self._artifact_data(artifact))
            for artifact in (getattr(info, "siblings", None) or [])
        ]
        model_data = summary.model_dump()
        model_data.update(
            config=config,
            card_data=card_data,
            siblings=siblings,
        )
        return HfModel.model_validate(model_data)

    def get_pipeline_requirements(self, model_id: str, include_config: bool = True) -> HfPipelineInfo:
        """Parse a diffusers `model_index.json` and match each declared component
        against the files actually present in the repo, so callers can see what
        a pipeline needs (text encoder, VAE, scheduler, ...) and whether it's there."""
        try:
            index_path = hf_hub_download(repo_id=model_id, filename=MODEL_INDEX_FILENAME)
        except EntryNotFoundError as error:
            raise ModelIndexNotFoundError(model_id) from error

        with open(index_path, encoding="utf-8") as file:
            index: dict[str, Any] = json.load(file)

        info = self.api.model_info(model_id, files_metadata=True)
        sibling_sizes = {
            sibling.rfilename: getattr(sibling, "size", None)
            for sibling in (getattr(info, "siblings", None) or [])
        }

        components: list[HfPipelineComponent] = []
        missing_components: list[str] = []
        extra_config: dict[str, Any] = {}

        for key, value in index.items():
            if key.startswith("_"):
                continue
            if not isinstance(value, list):
                extra_config[key] = value
                continue

            library, class_name = (list(value) + [None, None])[:2]
            required = class_name is not None

            prefix = f"{key}/"
            files = [
                HfPipelineFile(filename=name[len(prefix):], size=size)
                for name, size in sibling_sizes.items()
                if name.startswith(prefix)
            ]
            formats = sorted(
                {
                    WEIGHT_FORMATS_BY_EXTENSION[self._extension(pipeline_file.filename)]
                    for pipeline_file in files
                    if self._extension(pipeline_file.filename) in WEIGHT_FORMATS_BY_EXTENSION
                }
            )

            config_data = (
                self._download_component_config(model_id, key, files) if include_config else None
            )

            components.append(
                HfPipelineComponent(
                    name=key,
                    library=library,
                    class_name=class_name,
                    required=required,
                    subfolder=key,
                    files=files,
                    formats=formats,
                    config=config_data,
                )
            )

            if required and not files:
                missing_components.append(key)

        return HfPipelineInfo(
            model_id=getattr(info, "id", None) or getattr(info, "modelId", model_id),
            class_name=index.get("_class_name"),
            diffusers_version=index.get("_diffusers_version"),
            components=components,
            missing_components=missing_components,
            extra_config=extra_config,
        )

    @staticmethod
    def _download_component_config(
        model_id: str, subfolder: str, files: list[HfPipelineFile]
    ) -> dict[str, Any] | None:
        available = {pipeline_file.filename for pipeline_file in files}
        config_name = next((name for name in COMPONENT_CONFIG_FILENAMES if name in available), None)
        if config_name is None:
            return None
        try:
            config_path = hf_hub_download(repo_id=model_id, filename=f"{subfolder}/{config_name}")
            with open(config_path, encoding="utf-8") as file:
                return json.load(file)
        except (HfHubHTTPError, OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _extension(filename: str) -> str:
        return os.path.splitext(filename)[1].lower()

    @staticmethod
    def _summary_from_info(info: Any) -> HfModel:
        data = {
            "model_id": getattr(info, "id", None) or getattr(info, "modelId", ""),
            "author": getattr(info, "author", None),
            "sha": getattr(info, "sha", None),
            "downloads": getattr(info, "downloads", 0) or 0,
            "likes": getattr(info, "likes", 0) or 0,
            "trending_score": getattr(info, "trending_score", None),
            "private": getattr(info, "private", False) or False,
            "gated": getattr(info, "gated", False) or False,
            "disabled": getattr(info, "disabled", False) or False,
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": getattr(info, "library_name", None),
            "tags": getattr(info, "tags", None) or [],
            "languages": getattr(info, "languages", None) or [],
            "datasets": getattr(info, "datasets", None) or [],
            "last_modified": getattr(info, "last_modified", None),
            "created_at": getattr(info, "created_at", None),
        }
        return HfModel.model_validate(data)

    @staticmethod
    def _metadata_dict(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        if hasattr(value, "__dict__"):
            return vars(value)
        return None

    @staticmethod
    def _artifact_data(artifact: Any) -> dict[str, Any]:
        if hasattr(artifact, "model_dump"):
            data = artifact.model_dump()
        elif hasattr(artifact, "__dict__"):
            data = vars(artifact)
        else:
            data = {"filename": str(artifact)}
        data.setdefault("filename", "")
        if "size" not in data and "size_bytes" in data:
            data["size"] = data["size_bytes"]
        return data
