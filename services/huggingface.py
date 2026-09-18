from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from models.hf_model import (
    HfModel,
    HfModelArtifact,
    HfModelRequirements,
    HfPipelineComponent,
    HfPipelineFile,
    HfPipelineInfo,
    HfRequirementRole,
    HfSuggestedSource,
)
from schemas.hf_model import HfModelSearchRequest
from services.pipeline_registry import (
    FamilyDef,
    match_family_by_class,
    match_family_by_keywords,
    match_filename_alias,
    match_filename_alias_in_family,
)
from services.tensor_signatures import identify_by_tensor_names

MODEL_INDEX_FILENAME = "model_index.json"

WEIGHT_FORMATS_BY_EXTENSION = {
    ".safetensors": "safetensors",
    ".bin": "pytorch",
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".ckpt": "pytorch",
    ".msgpack": "flax",
    ".onnx": "onnx",
    ".onnx_data": "onnx",
    ".xml": "openvino",
    ".gguf": "gguf",
}

# Tensor-name fingerprinting reads a remote file's safetensors header (a range
# request, not a full download) - only that format exposes tensor names this
# cheaply, so it's the only one sniffed. GGUF has an equivalent header but
# parsing it is deliberately out of scope for now.
FINGERPRINT_EXTENSIONS = (".safetensors",)

COMPONENT_CONFIG_FILENAMES = (
    "config.json",
    "scheduler_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)

# `base_model:<relation>:<repo_id>` tag forms the Hub uses for derived repos
# (quantizations, finetunes, LoRA adapters, merges).
BASE_MODEL_RELATION_PREFIXES = ("quantized:", "finetune:", "adapter:", "merge:")

MAX_BASE_MODEL_CHAIN_DEPTH = 3
MAX_FINGERPRINT_FILES = 6

load_dotenv()


class ModelIndexNotFoundError(Exception):
    """Raised when a repo has no `model_index.json` (not a diffusers pipeline)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"{model_id} does not provide a model_index.json")


class HuggingFaceService:
    def __init__(self, api: HfApi | None = None) -> None:
        self.token = os.getenv("HF_TOKEN")
        self.api = api or HfApi(token=self.token)

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
            index_path = hf_hub_download(
                repo_id=model_id, filename=MODEL_INDEX_FILENAME, token=self.token
            )
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
            if not self._is_component_descriptor(value):
                # A real component is always `[library, class_name]` (both
                # strings, or both null when the component is disabled). Any
                # other list is pipeline config that happens to be list-shaped,
                # e.g. Krea2Pipeline's `text_encoder_select_layers: [2, 5, ...]`.
                extra_config[key] = value
                continue

            library, class_name = value
            required = class_name is not None

            prefix = f"{key}/"
            files = [
                HfPipelineFile(filename=name[len(prefix):], size=size)
                for name, size in sibling_sizes.items()
                if name.startswith(prefix)
            ]
            formats = self._formats_for_files(files)

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

    def _download_component_config(
        self, model_id: str, subfolder: str, files: list[HfPipelineFile]
    ) -> dict[str, Any] | None:
        available = {pipeline_file.filename for pipeline_file in files}
        config_name = next((name for name in COMPONENT_CONFIG_FILENAMES if name in available), None)
        if config_name is None:
            return None
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename=f"{subfolder}/{config_name}",
                token=self.token,
            )
            with open(config_path, encoding="utf-8") as file:
                return json.load(file)
        except (HfHubHTTPError, OSError, json.JSONDecodeError):
            return None

    def get_model_requirements(
        self,
        model_id: str,
        resolve_base_model: bool = True,
        fingerprint: bool = True,
        max_fingerprint_files: int = MAX_FINGERPRINT_FILES,
    ) -> HfModelRequirements:
        """Best-effort answer to "what does this model need to run" for repos that
        don't ship a `model_index.json` (GGUF quantizations, bare component repos,
        single-file checkpoints, ...). Falls back through the repo's `base_model`
        ancestry, then weight tensor-name fingerprinting, then a small curated
        table of known pipeline families."""
        try:
            pipeline = self.get_pipeline_requirements(model_id, include_config=False)
            return self._requirements_from_direct_pipeline(pipeline)
        except ModelIndexNotFoundError:
            pass

        info = self.api.model_info(model_id, files_metadata=True)
        target_filenames = [sibling.rfilename for sibling in (getattr(info, "siblings", None) or [])]
        notes: list[str] = []

        resolved_pipeline: HfPipelineInfo | None = None
        resolved_from: str | None = None
        if resolve_base_model:
            resolved_pipeline, resolved_from, chain_notes = self._resolve_from_base_model(model_id, info)
            notes.extend(chain_notes)

        if resolved_pipeline is not None:
            return self._requirements_from_ancestor_pipeline(
                model_id, resolved_pipeline, resolved_from, target_filenames, info, notes
            )

        return self._requirements_from_fingerprint(
            model_id, info, target_filenames, fingerprint, max_fingerprint_files, notes
        )

    def _requirements_from_direct_pipeline(self, pipeline: HfPipelineInfo) -> HfModelRequirements:
        family = match_family_by_class(pipeline.class_name)
        roles: list[HfRequirementRole] = []
        for component in pipeline.components:
            present = bool(component.files)
            sources: list[HfSuggestedSource] = []
            if component.required and not present:
                sources = self._registry_sources_for_role(family, component.name)
            roles.append(
                HfRequirementRole(
                    role=component.name,
                    architecture=component.class_name,
                    required=component.required,
                    present=present,
                    source="repo",
                    subfolder=component.subfolder,
                    files=[pipeline_file.filename for pipeline_file in component.files],
                    sources=sources,
                )
            )
        return HfModelRequirements(
            model_id=pipeline.model_id,
            family=family.name if family else None,
            pipeline_class=pipeline.class_name,
            resolved_from=None,
            roles=roles,
        )

    @staticmethod
    def _extract_base_model_ids(info: Any) -> list[str]:
        """Reads the Hub's own structured relation metadata: `base_model` in the
        card data, and `base_model:<relation>:<repo_id>` tags (relation is one of
        quantized/finetune/adapter/merge). This is how a repo like
        `city96/FLUX.1-dev-gguf` declares it's derived from
        `black-forest-labs/FLUX.1-dev` without any diffusers-specific metadata."""
        ids: list[str] = []
        card_data = getattr(info, "card_data", None) or getattr(info, "cardData", None)
        raw = getattr(card_data, "base_model", None) if card_data is not None else None
        if isinstance(raw, str):
            ids.append(raw)
        elif isinstance(raw, list):
            ids.extend(item for item in raw if isinstance(item, str))

        for tag in getattr(info, "tags", None) or []:
            if not tag.startswith("base_model:"):
                continue
            remainder = tag[len("base_model:") :]
            for prefix in BASE_MODEL_RELATION_PREFIXES:
                if remainder.startswith(prefix):
                    remainder = remainder[len(prefix) :]
                    break
            ids.append(remainder)

        seen: set[str] = set()
        ordered: list[str] = []
        for repo_id in ids:
            if repo_id and repo_id not in seen:
                seen.add(repo_id)
                ordered.append(repo_id)
        return ordered

    def _coarse_pipeline_from_metadata(self, model_id: str) -> HfPipelineInfo | None:
        """Fallback for gated base models: `hf_hub_download` needs auth, but the
        public `model_info` listing (tags + file tree) usually still works even
        without one. A `diffusers:<PipelineClass>` tag plus subfolder names
        (which, by diffusers convention, are exactly the model_index.json
        component keys) is enough to describe the pipeline shape without ever
        reading the JSON - at the cost of not knowing which components are
        optional, since that only lives inside the JSON itself."""
        try:
            info = self.api.model_info(model_id, files_metadata=True)
        except HfHubHTTPError:
            return None

        class_name = next(
            (
                tag.split(":", 1)[1]
                for tag in (getattr(info, "tags", None) or [])
                if tag.startswith("diffusers:")
            ),
            None,
        )
        if class_name is None:
            return None

        grouped: dict[str, list[HfPipelineFile]] = {}
        for sibling in getattr(info, "siblings", None) or []:
            subfolder, separator, filename = sibling.rfilename.partition("/")
            if not separator:
                continue
            grouped.setdefault(subfolder, []).append(
                HfPipelineFile(filename=filename, size=getattr(sibling, "size", None))
            )

        components = [
            HfPipelineComponent(
                name=subfolder,
                library=None,
                class_name=None,
                required=True,
                subfolder=subfolder,
                files=files,
                formats=self._formats_for_files(files),
            )
            for subfolder, files in grouped.items()
        ]
        return HfPipelineInfo(
            model_id=getattr(info, "id", None) or model_id,
            class_name=class_name,
            diffusers_version=None,
            components=components,
        )

    def _resolve_from_base_model(
        self, model_id: str, info: Any, max_depth: int = MAX_BASE_MODEL_CHAIN_DEPTH
    ) -> tuple[HfPipelineInfo | None, str | None, list[str]]:
        notes: list[str] = []
        visited = {model_id}
        frontier = self._extract_base_model_ids(info)

        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            next_frontier: list[str] = []
            for candidate in frontier:
                if candidate in visited:
                    continue
                visited.add(candidate)

                try:
                    pipeline = self.get_pipeline_requirements(candidate, include_config=False)
                    return pipeline, candidate, notes
                except ModelIndexNotFoundError:
                    pass
                except HfHubHTTPError as error:
                    status = getattr(getattr(error, "response", None), "status_code", None)
                    if status in (401, 403):
                        coarse = self._coarse_pipeline_from_metadata(candidate)
                        if coarse is not None:
                            notes.append(
                                f"base model {candidate} is gated; used its public file listing "
                                "instead of downloading model_index.json, so optional components "
                                "could not be distinguished from required ones"
                            )
                            return coarse, candidate, notes
                        notes.append(f"base model {candidate} is gated/private and inaccessible")
                    else:
                        notes.append(f"base model {candidate} could not be fetched ({status or 'error'})")
                    continue

                try:
                    candidate_info = self.api.model_info(candidate, files_metadata=False)
                    next_frontier.extend(self._extract_base_model_ids(candidate_info))
                except HfHubHTTPError:
                    pass
            frontier = next_frontier

        return None, None, notes

    def _requirements_from_ancestor_pipeline(
        self,
        model_id: str,
        pipeline: HfPipelineInfo,
        resolved_from: str,
        target_filenames: list[str],
        info: Any,
        notes: list[str],
    ) -> HfModelRequirements:
        family = match_family_by_class(pipeline.class_name) or match_family_by_keywords(
            model_id, " ".join(getattr(info, "tags", None) or [])
        )
        roles: list[HfRequirementRole] = []

        for component in pipeline.components:
            own_files = self._filenames_with_prefix(target_filenames, f"{component.name}/")
            if not own_files:
                own_files = self._filenames_by_alias(target_filenames, family, component.name)

            if own_files:
                roles.append(
                    HfRequirementRole(
                        role=component.name,
                        architecture=component.class_name,
                        required=component.required,
                        present=True,
                        source="repo",
                        files=own_files,
                    )
                )
                continue

            sources: list[HfSuggestedSource] = []
            if component.files:
                sources.append(
                    HfSuggestedSource(
                        repo_id=resolved_from,
                        files=[pipeline_file.filename for pipeline_file in component.files],
                        note=f"shipped in the base model's {component.subfolder}/ folder",
                    )
                )
            sources.extend(self._registry_sources_for_role(family, component.name))

            roles.append(
                HfRequirementRole(
                    role=component.name,
                    architecture=component.class_name,
                    required=component.required,
                    present=False,
                    source="base_model" if component.files else "registry",
                    files=[],
                    sources=sources,
                )
            )

        return HfModelRequirements(
            model_id=model_id,
            family=family.name if family else None,
            pipeline_class=pipeline.class_name,
            resolved_from=resolved_from,
            roles=roles,
            notes=notes,
        )

    def _requirements_from_fingerprint(
        self,
        model_id: str,
        info: Any,
        target_filenames: list[str],
        fingerprint: bool,
        max_fingerprint_files: int,
        notes: list[str],
    ) -> HfModelRequirements:
        tags = getattr(info, "tags", None) or []
        family = match_family_by_keywords(model_id, " ".join(tags))

        found: dict[str, HfRequirementRole] = {}
        unidentified: list[str] = []

        weight_files = [
            name for name in target_filenames if self._extension(name) in WEIGHT_FORMATS_BY_EXTENSION
        ]
        fingerprint_candidates = (
            [name for name in weight_files if self._extension(name) in FINGERPRINT_EXTENSIONS]
            if fingerprint
            else []
        )

        resolved: set[str] = set()
        for filename in fingerprint_candidates[:max_fingerprint_files]:
            try:
                metadata = self.api.parse_safetensors_file_metadata(model_id, filename)
            except (HfHubHTTPError, OSError) as error:
                notes.append(f"could not read tensor metadata for {filename} ({error})")
                continue

            identified = identify_by_tensor_names(list(metadata.tensors.keys()))
            if identified is None:
                continue
            role_name, architecture, confidence = identified
            resolved.add(filename)

            existing = self._find_role_with_architecture(found, architecture)
            if existing is not None:
                existing.files.append(filename)
                continue

            role_name = self._next_available_role_name(role_name, found)
            found[role_name] = HfRequirementRole(
                role=role_name,
                architecture=architecture,
                required=True,
                present=True,
                source="fingerprint",
                confidence=confidence,
                files=[filename],
            )

        for filename in weight_files:
            if filename in resolved:
                continue

            if family is not None:
                role_def = match_filename_alias_in_family(filename, family)
            else:
                match = match_filename_alias(filename)
                if match is not None:
                    family, role_def = match
                else:
                    role_def = None

            if role_def is None:
                unidentified.append(filename)
                continue

            existing = found.get(role_def.role)
            if existing is not None:
                existing.files.append(filename)
                continue

            found[role_def.role] = HfRequirementRole(
                role=role_def.role,
                architecture=role_def.architecture,
                required=role_def.required,
                present=True,
                source="repo",
                files=[filename],
            )

        if family is not None:
            for role_def in family.roles:
                if role_def.role in found:
                    continue
                found[role_def.role] = HfRequirementRole(
                    role=role_def.role,
                    architecture=role_def.architecture,
                    required=role_def.required,
                    present=False,
                    source="registry",
                    sources=self._registry_sources_for_role(family, role_def.role),
                )

        return HfModelRequirements(
            model_id=getattr(info, "id", None) or model_id,
            family=family.name if family else None,
            pipeline_class=None,
            resolved_from=None,
            roles=list(found.values()),
            unidentified_files=unidentified,
            notes=notes,
        )

    @staticmethod
    def _next_available_role_name(base_name: str, found: dict[str, HfRequirementRole]) -> str:
        if base_name not in found:
            return base_name
        index = 2
        while f"{base_name}_{index}" in found:
            index += 1
        return f"{base_name}_{index}"

    @staticmethod
    def _find_role_with_architecture(
        found: dict[str, HfRequirementRole], architecture: str | None
    ) -> HfRequirementRole | None:
        if architecture is None:
            return None
        return next((role for role in found.values() if role.architecture == architecture), None)

    @staticmethod
    def _filenames_with_prefix(filenames: list[str], prefix: str) -> list[str]:
        return [name[len(prefix) :] for name in filenames if name.startswith(prefix)]

    @staticmethod
    def _filenames_by_alias(filenames: list[str], family: FamilyDef | None, role_name: str) -> list[str]:
        aliases: tuple[str, ...] = (role_name,)
        if family is not None:
            role_def = next((role for role in family.roles if role.role == role_name), None)
            if role_def is not None:
                aliases = role_def.filename_aliases
        matches = []
        for name in filenames:
            basename = name.rsplit("/", 1)[-1].lower()
            if any(alias in basename for alias in aliases):
                matches.append(name)
        return matches

    @staticmethod
    def _registry_sources_for_role(family: FamilyDef | None, role_name: str) -> list[HfSuggestedSource]:
        if family is None:
            return []
        role_def = next((role for role in family.roles if role.role == role_name), None)
        if role_def is None or role_def.companion_repo is None:
            return []
        return [HfSuggestedSource(repo_id=role_def.companion_repo, files=list(role_def.companion_files))]

    @staticmethod
    def _is_component_descriptor(value: Any) -> bool:
        return (
            isinstance(value, list)
            and len(value) == 2
            and all(item is None or isinstance(item, str) for item in value)
        )

    @staticmethod
    def _extension(filename: str) -> str:
        return os.path.splitext(filename)[1].lower()

    @classmethod
    def _formats_for_files(cls, files: list[HfPipelineFile]) -> list[str]:
        return sorted(
            {
                WEIGHT_FORMATS_BY_EXTENSION[cls._extension(pipeline_file.filename)]
                for pipeline_file in files
                if cls._extension(pipeline_file.filename) in WEIGHT_FORMATS_BY_EXTENSION
            }
        )

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
