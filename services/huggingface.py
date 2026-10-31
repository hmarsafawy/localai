from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

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
from services.tensor_signatures import (
    HIGH_CONFIDENCE_COVERAGE,
    ReferenceMatch,
    has_vision_tensors,
    identify_by_tensor_names,
    is_lora,
    match_reference,
)
from services.weight_headers import TensorShapes, read_tensor_shapes

MODEL_INDEX_FILENAME = "model_index.json"
# Modular diffusers pipelines ship this instead. Same component keys, but each
# entry is `[library, class_name, spec]`, and the spec's repo/subfolder can
# point outside the component's own folder - or at another repo entirely.
MODULAR_MODEL_INDEX_FILENAME = "modular_model_index.json"
PIPELINE_INDEX_FILENAMES = (MODEL_INDEX_FILENAME, MODULAR_MODEL_INDEX_FILENAME)

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

# Tensor-name fingerprinting reads a remote file's header (a range request, not
# a full download) - only these formats expose tensor names that cheaply.
FINGERPRINT_EXTENSIONS = (".safetensors", ".gguf")
SHARD_INDEX_SUFFIX = ".safetensors.index.json"

# llama.cpp's name for a multimodal projector file - distinctive enough to
# identify one by name when its header isn't read. The broader hints only move
# a file to the front of the header-read queue, so a repo with a dozen LLM
# quantizations doesn't spend the whole budget before reaching its mmproj.
MMPROJ_FILENAME_HINT = "mmproj"
VISION_FILENAME_HINTS = (MMPROJ_FILENAME_HINT, "vision", "visual", "projector", "siglip")
# Roles a fingerprinted LLM is a text encoder for rather than a model in its own right.
DIFFUSION_ROLES = ("transformer", "unet", "vae")
MAX_VISION_CONFIG_REPOS = 2

# transformers' naming convention for weightless preprocessing classes
# (CLIPTokenizer, T5TokenizerFast, AutoProcessor, CLIPImageProcessor, ...).
# Modular indexes point these at the same folder as their model, and they must
# not claim that model's weights as their own files.
PREPROCESSOR_CLASS_SUFFIXES = ("Tokenizer", "TokenizerFast", "Processor", "FeatureExtractor")

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
# Repos like ComfyUI/GGUF bundles ship a dozen quantizations of one component;
# headers are tiny, so reading all of them is cheap, but keep a ceiling.
MAX_REPO_MATCH_FILES = 32
MAX_HEADER_READ_WORKERS = 8
# Bundled checkpoints nest components up to ~3 segments deep
# (`text_encoders.clip_l.transformer.`); deeper splits are just layer names.
MAX_BUNDLE_PREFIX_DEPTH = 3

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")
# (role, repo, its weight files' repo paths, headers already read from that repo)
_VisionCheck = tuple[HfRequirementRole, str, list[str], dict[str, TensorShapes]]

load_dotenv()


class ModelIndexNotFoundError(Exception):
    """Raised when a repo has no `model_index.json` or `modular_model_index.json`
    (not a diffusers pipeline)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"{model_id} does not provide a model_index.json or modular_model_index.json")


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
        """Parse a diffusers `model_index.json` (or a modular pipeline's
        `modular_model_index.json`) and match each declared component against
        the files actually present, so callers can see what a pipeline needs
        (text encoder, VAE, scheduler, ...) and whether it's there."""
        index = self._download_pipeline_index(model_id)

        info = self.api.model_info(model_id, files_metadata=True)
        resolved_id = getattr(info, "id", None) or getattr(info, "modelId", model_id)
        own_repo_ids = {model_id.lower(), resolved_id.lower()}
        sizes_by_repo: dict[str, dict[str, int | None]] = {resolved_id: self._sibling_sizes(info)}

        components: list[HfPipelineComponent] = []
        missing_components: list[str] = []
        extra_config: dict[str, Any] = {}

        for key, value in index.items():
            if key.startswith("_"):
                continue
            if not self._is_component_descriptor(value):
                # A real component is always `[library, class_name]` (both
                # strings, or both null when the component is disabled), plus
                # a spec dict in modular indexes. Any other list is pipeline
                # config that happens to be list-shaped, e.g. Krea2Pipeline's
                # `text_encoder_select_layers: [2, 5, ...]`.
                extra_config[key] = value
                continue

            library, class_name = value[0], value[1]
            source_repo: str | None = resolved_id
            subfolder = key
            if len(value) == 3:
                # A spec with no repo is built from config at load time, so
                # there's nothing to download for it.
                source_repo = value[2].get("pretrained_model_name_or_path")
                subfolder = value[2].get("subfolder") or ""
                if source_repo is not None and source_repo.lower() in own_repo_ids:
                    source_repo = resolved_id
            required = class_name is not None and source_repo is not None

            files: list[HfPipelineFile] = []
            if source_repo is not None:
                if source_repo not in sizes_by_repo:
                    sizes_by_repo[source_repo] = self._external_sibling_sizes(source_repo)
                files = self._files_in_subfolder(sizes_by_repo[source_repo], subfolder)
            if class_name is not None and class_name.endswith(PREPROCESSOR_CLASS_SUFFIXES):
                files = [
                    pipeline_file
                    for pipeline_file in files
                    if self._extension(pipeline_file.filename) not in WEIGHT_FORMATS_BY_EXTENSION
                    and not pipeline_file.filename.endswith(SHARD_INDEX_SUFFIX)
                ]
            external_repo = source_repo if source_repo not in (None, resolved_id) else None

            config_data = (
                self._download_component_config(source_repo, subfolder, files)
                if include_config and source_repo is not None
                else None
            )

            components.append(
                HfPipelineComponent(
                    name=key,
                    library=library,
                    class_name=class_name,
                    required=required,
                    subfolder=subfolder,
                    repo_id=external_repo,
                    files=files,
                    formats=self._formats_for_files(files),
                    config=config_data,
                )
            )

            if required and not files:
                missing_components.append(key)

        return HfPipelineInfo(
            model_id=resolved_id,
            class_name=index.get("_class_name"),
            diffusers_version=index.get("_diffusers_version"),
            components=components,
            missing_components=missing_components,
            extra_config=extra_config,
        )

    def _download_pipeline_index(self, model_id: str) -> dict[str, Any]:
        for filename in PIPELINE_INDEX_FILENAMES:
            try:
                index_path = hf_hub_download(repo_id=model_id, filename=filename, token=self.token)
            except EntryNotFoundError:
                continue
            with open(index_path, encoding="utf-8") as file:
                return json.load(file)
        raise ModelIndexNotFoundError(model_id)

    @staticmethod
    def _sibling_sizes(info: Any) -> dict[str, int | None]:
        return {
            sibling.rfilename: getattr(sibling, "size", None)
            for sibling in (getattr(info, "siblings", None) or [])
        }

    def _external_sibling_sizes(self, repo_id: str) -> dict[str, int | None]:
        # A missing/gated component repo shouldn't sink the whole pipeline; it
        # just shows up as a component with no files. ValueError covers a spec
        # that names a local path instead of a repo id.
        try:
            return self._sibling_sizes(self.api.model_info(repo_id, files_metadata=True))
        except (HfHubHTTPError, ValueError):
            return {}

    @staticmethod
    def _files_in_subfolder(sizes: dict[str, int | None], subfolder: str) -> list[HfPipelineFile]:
        # An empty subfolder means the repo root: only its top-level files, not
        # everything nested under it.
        if not subfolder:
            return [HfPipelineFile(filename=name, size=size) for name, size in sizes.items() if "/" not in name]
        prefix = f"{subfolder}/"
        return [
            HfPipelineFile(filename=name[len(prefix) :], size=size)
            for name, size in sizes.items()
            if name.startswith(prefix)
        ]

    @staticmethod
    def _repo_path(subfolder: str, filename: str) -> str:
        return f"{subfolder}/{filename}" if subfolder else filename

    def _download_component_config(
        self, repo_id: str, subfolder: str, files: list[HfPipelineFile]
    ) -> dict[str, Any] | None:
        available = {pipeline_file.filename for pipeline_file in files}
        config_name = next((name for name in COMPONENT_CONFIG_FILENAMES if name in available), None)
        if config_name is None:
            return None
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=self._repo_path(subfolder, config_name),
                token=self.token,
            )
            with open(config_path, encoding="utf-8") as file:
                return json.load(file)
        except (HfHubHTTPError, OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _class_from_config(config: dict[str, Any] | None) -> tuple[str, str] | None:
        """(library, class name) a component's own config declares: diffusers
        writes `_class_name`; transformers writes `architectures` for models and
        `tokenizer_class`/`processor_class`/`image_processor_type` for the rest."""
        if not config:
            return None
        if isinstance(config.get("_class_name"), str):
            return "diffusers", config["_class_name"]
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures and isinstance(architectures[0], str):
            return "transformers", architectures[0]
        for key in ("tokenizer_class", "processor_class", "image_processor_type"):
            if isinstance(config.get(key), str):
                return "transformers", config[key]
        return None

    def get_model_requirements(
        self,
        model_id: str,
        resolve_base_model: bool = True,
        fingerprint: bool = True,
        max_fingerprint_files: int = MAX_FINGERPRINT_FILES,
    ) -> HfModelRequirements:
        """Best-effort answer to "what does this model need to run" for repos that
        don't ship a pipeline index (GGUF quantizations, bare component repos,
        single-file checkpoints, ...). Falls back through the repo's `base_model`
        ancestry, then the per-subfolder configs of a diffusers-layout repo, then
        weight tensor-name fingerprinting plus a small curated table of known
        pipeline families."""
        try:
            pipeline = self.get_pipeline_requirements(model_id, include_config=False)
            return self._requirements_from_direct_pipeline(pipeline)
        except ModelIndexNotFoundError:
            pass
        except HfHubHTTPError as error:
            # Gated repo whose license the token hasn't accepted: the file
            # listing is still public, which is enough for the pipeline shape.
            status = getattr(getattr(error, "response", None), "status_code", None)
            coarse = self._coarse_pipeline_from_metadata(model_id) if status in (401, 403) else None
            if coarse is None:
                raise
            return self._requirements_from_direct_pipeline(
                coarse,
                notes=[
                    f"{model_id} is gated and its license hasn't been accepted for this token; used its "
                    "public file listing, so component classes and optional components are unknown"
                ],
            )

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
                model_id, resolved_pipeline, resolved_from, target_filenames, info, fingerprint, notes
            )

        # The repo's own component configs say exactly what each part is, which
        # beats guessing a family from its name - "flux" in a repo id says
        # nothing about which generation's text encoders it uses.
        self_described = self._pipeline_from_component_configs(model_id, info)
        if self_described is not None:
            notes.append("no pipeline index; components were read from each subfolder's own config")
            return self._requirements_from_ancestor_pipeline(
                model_id, self_described, None, target_filenames, info, fingerprint, notes
            )

        return self._requirements_from_fingerprint(
            model_id, info, target_filenames, fingerprint, max_fingerprint_files, notes
        )

    def _requirements_from_direct_pipeline(
        self, pipeline: HfPipelineInfo, notes: list[str] | None = None
    ) -> HfModelRequirements:
        family = match_family_by_class(pipeline.class_name)
        roles: list[HfRequirementRole] = []
        vision_checks: list[_VisionCheck] = []
        for component in pipeline.components:
            files = [pipeline_file.filename for pipeline_file in component.files]
            repo_paths = [self._repo_path(component.subfolder, name) for name in files]
            if component.repo_id is not None:
                role = HfRequirementRole(
                    role=component.name,
                    architecture=component.class_name,
                    required=component.required,
                    present=False,
                    source="external",
                    sources=[
                        HfSuggestedSource(
                            repo_id=component.repo_id,
                            files=repo_paths,
                            note="the pipeline index loads this component from another repo",
                        )
                    ],
                )
                roles.append(role)
                if self._is_text_encoder_role(component.name) and files:
                    vision_checks.append((role, component.repo_id, repo_paths, {}))
                continue

            present = bool(files)
            sources: list[HfSuggestedSource] = []
            if component.required and not present:
                sources = self._registry_sources_for_role(family, component.name)
            role = HfRequirementRole(
                role=component.name,
                architecture=component.class_name,
                required=component.required,
                present=present,
                source="repo",
                subfolder=component.subfolder or None,
                files=files,
                sources=sources,
            )
            roles.append(role)
            if self._is_text_encoder_role(component.name) and present:
                vision_checks.append((role, pipeline.model_id, repo_paths, {}))

        self._mark_vision_weights(vision_checks)
        return HfModelRequirements(
            model_id=pipeline.model_id,
            family=family.name if family else None,
            pipeline_class=pipeline.class_name,
            resolved_from=None,
            roles=roles,
            notes=notes or [],
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
            for subfolder, files in self._files_by_subfolder(info).items()
        ]
        return HfPipelineInfo(
            model_id=getattr(info, "id", None) or model_id,
            class_name=class_name,
            diffusers_version=None,
            components=components,
        )

    def _pipeline_from_component_configs(self, model_id: str, info: Any) -> HfPipelineInfo | None:
        """For a repo laid out the diffusers way (`transformer/`, `vae/`,
        `text_encoder/`, ...) that just never wrote a pipeline index: each
        subfolder's own config still names its class, which is all the index
        would have said. Only used when at least one subfolder is a diffusers
        component - a lone transformers folder next to, say, a single-file
        checkpoint isn't a pipeline layout, and treating it as one would leave
        that checkpoint to fingerprinting against the wrong reference."""
        candidates = [
            (subfolder, files)
            for subfolder, files in self._files_by_subfolder(info).items()
            if any(pipeline_file.filename in COMPONENT_CONFIG_FILENAMES for pipeline_file in files)
        ]
        configs = self._run_parallel(
            lambda candidate: self._download_component_config(model_id, *candidate), candidates
        )

        components: list[HfPipelineComponent] = []
        for (subfolder, files), config in zip(candidates, configs):
            declared = self._class_from_config(config)
            if declared is None:
                continue
            library, class_name = declared
            components.append(
                HfPipelineComponent(
                    name=subfolder,
                    library=library,
                    class_name=class_name,
                    required=True,
                    subfolder=subfolder,
                    files=files,
                    formats=self._formats_for_files(files),
                )
            )

        if not any(component.library == "diffusers" for component in components):
            return None
        return HfPipelineInfo(model_id=getattr(info, "id", None) or model_id, components=components)

    @staticmethod
    def _files_by_subfolder(info: Any) -> dict[str, list[HfPipelineFile]]:
        grouped: dict[str, list[HfPipelineFile]] = {}
        for sibling in getattr(info, "siblings", None) or []:
            subfolder, separator, filename = sibling.rfilename.partition("/")
            if not separator:
                continue
            grouped.setdefault(subfolder, []).append(
                HfPipelineFile(filename=filename, size=getattr(sibling, "size", None))
            )
        return grouped

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
        resolved_from: str | None,
        target_filenames: list[str],
        info: Any,
        fingerprint: bool,
        notes: list[str],
    ) -> HfModelRequirements:
        # No `resolved_from` means the pipeline was read from this repo itself,
        # so its loose files are compared against its own components.
        reference_repo = resolved_from or model_id
        family = match_family_by_class(pipeline.class_name)
        if family is None:
            # A name keyword only stands in for an unlisted pipeline class if
            # the pipeline's own components belong to that family - "flux" in
            # a FLUX.3 repo must not pull in FLUX.1's filename aliases.
            guess = match_family_by_keywords(model_id, " ".join(getattr(info, "tags", None) or []))
            if guess is not None and any(
                self._family_has_architecture(guess, component.class_name) for component in pipeline.components
            ):
                family = guess
        roles: list[HfRequirementRole] = []

        # Full repo paths claimed per component by folder name or filename alias.
        own_files_by_component: dict[str, list[str]] = {}
        for component in pipeline.components:
            prefix = f"{component.name}/"
            own_files = [name for name in target_filenames if name.startswith(prefix)]
            if not own_files:
                own_files = self._filenames_by_alias(target_filenames, family, component.name)
            own_files_by_component[component.name] = own_files
        claimed = {name for files in own_files_by_component.values() for name in files}
        in_component_folders = {
            name
            for component in pipeline.components
            for name in target_filenames
            if name.startswith(f"{component.name}/")
        }

        # Every weight file outside the components' own folders is compared by
        # tensor names against the base model's components, so the repo's
        # copies win before the base model is suggested. That includes files a
        # filename alias already placed: a name only says what a file is
        # *mainly*, and an all-in-one checkpoint named after its transformer
        # (`flux1-dev-fp8.safetensors`) also carries the text encoders and VAE.
        matched: dict[str, list[tuple[str, ReferenceMatch]]] = {}
        lora_files: list[str] = []
        unidentified: list[str] = []
        headers: dict[str, TensorShapes] = {}
        if fingerprint:
            candidates = [
                name
                for name in target_filenames
                if name not in in_component_folders and self._extension(name) in FINGERPRINT_EXTENSIONS
            ]
            if candidates:
                matched, lora_files, unidentified, headers = self._match_files_to_components(
                    model_id, pipeline, reference_repo, candidates, notes
                )
                unidentified = [name for name in unidentified if name not in claimed]
                lora_files = [name for name in lora_files if name not in claimed]

        components_by_file: dict[str, list[str]] = {}
        for component_name, files in own_files_by_component.items():
            for filename in files:
                components_by_file.setdefault(filename, []).append(component_name)
        for component_name, files in matched.items():
            for filename, _ in files:
                if component_name not in components_by_file.setdefault(filename, []):
                    components_by_file[filename].append(component_name)
        for filename, component_names in components_by_file.items():
            if len(component_names) > 1:
                notes.append(
                    f"{filename} bundles several components in one file ({', '.join(component_names)})"
                )

        vision_checks: list[_VisionCheck] = []
        for component in pipeline.components:
            own_files = own_files_by_component[component.name]
            fingerprinted = [
                (filename, match) for filename, match in matched.get(component.name, []) if filename not in own_files
            ]

            if own_files or fingerprinted:
                prefix = f"{component.name}/"
                only_subfolder = not fingerprinted and all(name.startswith(prefix) for name in own_files)
                files = own_files + [filename for filename, _ in fingerprinted]
                confidence = "high"
                if not own_files and all(match.confidence == "low" for _, match in fingerprinted):
                    confidence = "low"
                role = HfRequirementRole(
                    role=component.name,
                    architecture=component.class_name,
                    required=component.required,
                    present=True,
                    source="repo" if own_files else "fingerprint",
                    confidence=confidence,
                    subfolder=component.name if only_subfolder else None,
                    files=[name[len(prefix) :] for name in files] if only_subfolder else files,
                )
                roles.append(role)
                if self._is_text_encoder_role(component.name):
                    vision_checks.append((role, model_id, files, headers))
                full_copies = [
                    filename
                    for filename, match in fingerprinted
                    if match.reference_coverage >= HIGH_CONFIDENCE_COVERAGE
                ]
                if len(full_copies) > 1:
                    notes.append(
                        f"{component.name}: {len(full_copies)} files each hold the complete component "
                        "(different precisions/quantizations/variants) - only one of them is needed"
                    )
                converted = [filename for filename, match in fingerprinted if match.confidence == "low"]
                if converted:
                    notes.append(
                        f"{component.name}: {len(converted)} file(s) only partially match {reference_repo}'s "
                        "tensor names (renamed, fused or repacked layers) - likely converted copies, e.g. "
                        "ComfyUI/native format, that need a loader for that format: " + ", ".join(converted)
                    )
                continue

            sources: list[HfSuggestedSource] = []
            if component.files:
                sources.append(
                    HfSuggestedSource(
                        repo_id=component.repo_id or reference_repo,
                        files=[
                            self._repo_path(component.subfolder, pipeline_file.filename)
                            for pipeline_file in component.files
                        ],
                        note=(
                            "the pipeline index loads this component from another repo"
                            if component.repo_id
                            else "shipped in the base model"
                        ),
                    )
                )
            sources.extend(self._registry_sources_for_role(family, component.name))

            role = HfRequirementRole(
                role=component.name,
                architecture=component.class_name,
                required=component.required,
                present=False,
                source=(
                    "external" if component.repo_id else "base_model" if component.files else "registry"
                ),
                files=[],
                sources=sources,
            )
            roles.append(role)
            if self._is_text_encoder_role(component.name) and component.files:
                # Describes the copy the first suggested source would bring in.
                vision_checks.append((role, component.repo_id or reference_repo, sources[0].files, {}))

        self._mark_vision_weights(vision_checks)

        if lora_files:
            roles.append(
                HfRequirementRole(
                    role="lora",
                    architecture=None,
                    required=False,
                    present=True,
                    source="fingerprint",
                    files=lora_files,
                )
            )

        return HfModelRequirements(
            model_id=model_id,
            family=family.name if family else None,
            pipeline_class=pipeline.class_name,
            resolved_from=resolved_from,
            roles=roles,
            unidentified_files=unidentified,
            notes=notes,
        )

    def _match_files_to_components(
        self,
        model_id: str,
        pipeline: HfPipelineInfo,
        reference_repo: str,
        candidates: list[str],
        notes: list[str],
    ) -> tuple[dict[str, list[tuple[str, ReferenceMatch]]], list[str], list[str], dict[str, TensorShapes]]:
        """Attribute each candidate weight file in `model_id` to whichever of the
        reference pipeline's components it matches best by tensor names/shapes.
        Returns (component name -> [(file, match)], LoRA files, unmatched files,
        the headers read)."""
        references = self._component_references(reference_repo, pipeline, notes)
        if not references:
            return {}, [], list(candidates), {}

        to_read = candidates[:MAX_REPO_MATCH_FILES]
        unidentified = list(candidates[MAX_REPO_MATCH_FILES:])
        if unidentified:
            notes.append(
                f"only the first {MAX_REPO_MATCH_FILES} weight files were fingerprinted; "
                f"{len(unidentified)} were skipped"
            )

        headers = self._read_headers(model_id, to_read, notes)
        matched: dict[str, list[tuple[str, ReferenceMatch]]] = {}
        lora_files: list[str] = []
        unmatched: list[str] = []
        for filename in to_read:
            shapes = headers.get(filename)
            if not shapes:
                unidentified.append(filename)
                continue
            if is_lora(list(shapes)):
                lora_files.append(filename)
                continue

            found = self._match_tensor_groups(shapes, references)
            if not found:
                unmatched.append(filename)
                continue
            for component_name, match in found.items():
                matched.setdefault(component_name, []).append((filename, match))

        # Quantized siblings often can't be matched to the reference directly
        # (renamed scheme *and* changed shapes, or fused layers), but they keep
        # the naming of the full-precision file next to them. So compare what's
        # left against files this repo already resolved: a match inherits that
        # file's component, and only stays high-confidence if both links are.
        # Bundles are left out: matching one says nothing about which of its
        # components the file resembles.
        components_per_file = Counter(filename for files in matched.values() for filename, _ in files)
        peers = {
            filename: (component_name, match)
            for component_name, files in matched.items()
            for filename, match in files
            if components_per_file[filename] == 1
        }
        for filename in unmatched:
            best = self._best_reference_match(
                headers[filename],
                {peer: (list(headers[peer]), headers[peer]) for peer in peers},
            )
            if best is None:
                unidentified.append(filename)
                continue
            peer_name, peer_match = best
            component_name, reference_match = peers[peer_name]
            confident = peer_match.confidence == "high" and reference_match.confidence == "high"
            matched[component_name].append(
                (
                    filename,
                    ReferenceMatch(
                        peer_match.candidate_coverage,
                        min(peer_match.reference_coverage, reference_match.reference_coverage),
                        "high" if confident else "low",
                        f"{peer_match.method} via {peer_name}",
                    ),
                )
            )

        return matched, lora_files, unidentified, headers

    @classmethod
    def _match_tensor_groups(
        cls,
        shapes: TensorShapes,
        references: dict[str, tuple[list[str], TensorShapes | None]],
        depth: int = 0,
    ) -> dict[str, ReferenceMatch]:
        """Which reference components a file holds. Normally the whole file
        matches one; if not, it may be a bundle (ComfyUI/A1111 all-in-one
        checkpoints nest each part under its own prefix: `model.diffusion_model.`,
        `text_encoders.clip_l.`, `vae.`), so its tensors are split by leading
        name segment and each group is matched on its own, a few levels deep."""
        best = cls._best_reference_match(shapes, references)
        if best is not None:
            return {best[0]: best[1]}
        if depth >= MAX_BUNDLE_PREFIX_DEPTH:
            return {}

        groups: dict[str, TensorShapes] = {}
        for key, shape in shapes.items():
            head, separator, rest = key.partition(".")
            # A numbered segment is a layer index (`double_blocks.0.`), never a
            # component container, and splitting there is most of the cost.
            if separator and not head.isdigit():
                groups.setdefault(head, {})[rest] = shape
        found: dict[str, ReferenceMatch] = {}
        for group in groups.values():
            for component_name, match in cls._match_tensor_groups(group, references, depth + 1).items():
                if component_name not in found or match.score > found[component_name].score:
                    found[component_name] = match
        return found

    @staticmethod
    def _best_reference_match(
        shapes: TensorShapes, references: dict[str, tuple[list[str], TensorShapes | None]]
    ) -> tuple[str, ReferenceMatch] | None:
        best: tuple[str, ReferenceMatch] | None = None
        for name, (reference_keys, reference_shapes) in references.items():
            match = match_reference(shapes, reference_keys, reference_shapes)
            if match is not None and (best is None or match.score > best[1].score):
                best = (name, match)
        return best

    def _component_references(
        self, repo_id: str, pipeline: HfPipelineInfo, notes: list[str]
    ) -> dict[str, tuple[list[str], TensorShapes | None]]:
        """Tensor names and shapes of each weight-bearing component in a
        diffusers repo, read from every shard's header. A sharded component's
        `*.index.json` would list the names in one download, but without shapes
        a copy saved under another naming scheme (a ComfyUI/BFL-native
        transformer vs the diffusers one: same tensors, different names) can't
        be recognised, and that's exactly the copy derived repos tend to ship."""
        # Modular indexes can point two components at one folder (a tokenizer
        # sharing `text_encoder/`); its weights only need reading once.
        weight_files: list[tuple[str, str, str]] = []  # (component, repo, path)
        seen_locations: set[tuple[str, str]] = set()
        for component in pipeline.components:
            source_repo = component.repo_id or repo_id
            location = (source_repo, component.subfolder)
            if location in seen_locations:
                continue
            seen_locations.add(location)
            weight_files.extend(
                (component.name, source_repo, self._repo_path(component.subfolder, pipeline_file.filename))
                for pipeline_file in component.files
                if self._extension(pipeline_file.filename) in FINGERPRINT_EXTENSIONS
            )

        def read(weight_file: tuple[str, str, str]) -> TensorShapes | None:
            component_name, source_repo, path = weight_file
            try:
                return read_tensor_shapes(self.api, self.token, source_repo, path)
            except (HfHubHTTPError, OSError, ValueError) as error:
                notes.append(f"could not read {source_repo}'s {component_name} tensor list ({error})")
                return None

        shapes_by_component: dict[str, TensorShapes] = {}
        failed: set[str] = set()
        for (component_name, _, _), shapes in zip(weight_files, self._run_parallel(read, weight_files)):
            if shapes is None:
                # A reference missing a shard would make every full copy look
                # partial, so drop the component rather than compare against it.
                failed.add(component_name)
                continue
            shapes_by_component.setdefault(component_name, {}).update(shapes)
        return {
            name: (list(shapes), shapes)
            for name, shapes in shapes_by_component.items()
            if name not in failed and shapes
        }

    def _read_headers(self, repo_id: str, filenames: list[str], notes: list[str]) -> dict[str, TensorShapes]:
        def read(filename: str) -> TensorShapes | None:
            try:
                return read_tensor_shapes(self.api, self.token, repo_id, filename)
            except (HfHubHTTPError, OSError, ValueError) as error:
                notes.append(f"could not read tensor metadata for {filename} ({error})")
                return None

        return {
            filename: shapes
            for filename, shapes in zip(filenames, self._run_parallel(read, filenames))
            if shapes is not None
        }

    @staticmethod
    def _run_parallel(function: Callable[[_Item], _Result], items: list[_Item]) -> list[_Result]:
        if len(items) <= 1:
            return [function(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(MAX_HEADER_READ_WORKERS, len(items))) as executor:
            return list(executor.map(function, items))

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
        # A keyword hit ("flux" in "flux-3-...") can't tell a family's
        # generations apart, so on its own it's only a guess. A fingerprinted
        # architecture or a brand-specific filename from that family confirms it.
        family_confirmed = False
        registry_roles: set[str] = set()

        found: dict[str, HfRequirementRole] = {}
        unidentified: list[str] = []

        weight_files = [
            name for name in target_filenames if self._extension(name) in WEIGHT_FORMATS_BY_EXTENSION
        ]

        # Each unit is (files, tensor names), identified as a whole: a sharded
        # checkpoint by its index's full tensor list - a VL model's vision tower
        # usually sits in its last shard - and any other file by its own header.
        units: list[tuple[list[str], list[str]]] = []
        sharded: set[str] = set()
        if fingerprint:
            index_files = [name for name in target_filenames if name.endswith(SHARD_INDEX_SUFFIX)]
            for index_file, weight_map in self._read_weight_maps(model_id, index_files, notes).items():
                shards = sorted(set(weight_map.values()))
                sharded.update(shards)
                units.append(([index_file, *shards], list(weight_map)))

        fingerprint_candidates = (
            [
                name
                for name in weight_files
                if self._extension(name) in FINGERPRINT_EXTENSIONS and name not in sharded
            ]
            if fingerprint
            else []
        )
        fingerprint_candidates.sort(key=lambda name: not self._has_vision_hint(name))

        resolved: set[str] = set()
        to_read = fingerprint_candidates[:max_fingerprint_files]
        headers = self._read_headers(model_id, to_read, notes)
        units.extend(([filename], list(headers[filename])) for filename in to_read if filename in headers)
        for files, tensor_keys in units:
            identified = identify_by_tensor_names(tensor_keys)
            if identified is None:
                continue
            role_name, architecture, confidence = identified
            resolved.update(files)
            if family is not None and self._family_has_architecture(family, architecture):
                family_confirmed = True
            has_vision = (
                has_vision_tensors(tensor_keys)
                if role_name == "vision_encoder" or self._is_text_encoder_role(role_name)
                else None
            )

            existing = self._find_same_model_role(found, role_name, architecture)
            if existing is not None:
                existing.files.extend(files)
                if has_vision is not None:
                    existing.has_vision_weights = bool(existing.has_vision_weights) or has_vision
                continue

            role_name = self._next_available_role_name(role_name, found)
            found[role_name] = HfRequirementRole(
                role=role_name,
                architecture=architecture,
                required=True,
                present=True,
                source="fingerprint",
                confidence=confidence,
                files=list(files),
                has_vision_weights=has_vision,
            )

        if any(name in found for name in DIFFUSION_ROLES):
            # An LLM next to a diffusion model is that model's text encoder
            # (Qwen-Image's Qwen2.5-VL, Z-Image's Qwen3, ...).
            for name in [name for name in found if name.startswith("language_model")]:
                role = found.pop(name)
                role.role = self._next_available_role_name("text_encoder", found)
                found[role.role] = role
        is_language_model = any(name.startswith("language_model") for name in found)

        for filename in weight_files:
            if filename in resolved:
                continue

            # An mmproj whose header wasn't read (or didn't look vision-only,
            # e.g. an omni model's combined vision+audio projector) still
            # names itself; one that was read must actually carry vision weights.
            if MMPROJ_FILENAME_HINT in filename.rsplit("/", 1)[-1].lower() and (
                filename not in headers or has_vision_tensors(list(headers[filename]))
            ):
                existing = found.get("vision_encoder")
                if existing is not None:
                    existing.files.append(filename)
                else:
                    found["vision_encoder"] = HfRequirementRole(
                        role="vision_encoder",
                        present=True,
                        source="repo",
                        files=[filename],
                        has_vision_weights=True,
                    )
                continue

            discovered = match_filename_alias(filename)
            if family is None and discovered is not None:
                family = discovered[0]
            if discovered is not None and discovered[0] is family:
                family_confirmed = True
            role_def = match_filename_alias_in_family(filename, family) if family is not None else None

            if role_def is None:
                unidentified.append(filename)
                continue

            existing = found.get(role_def.role)
            if existing is not None:
                existing.files.append(filename)
                continue

            registry_roles.add(role_def.role)
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
                registry_roles.add(role_def.role)
                found[role_def.role] = HfRequirementRole(
                    role=role_def.role,
                    architecture=role_def.architecture,
                    required=role_def.required,
                    present=False,
                    source="registry",
                    sources=self._registry_sources_for_role(family, role_def.role),
                )
            if not family_confirmed and registry_roles:
                for role_name in registry_roles:
                    found[role_name].confidence = "low"
                notes.append(
                    f"the '{family.name}' family was guessed from the repo name/tags only - no weight file "
                    "confirms it, so its expected roles and architectures may belong to a different "
                    "generation of the model"
                )

        if is_language_model:
            # An LLM runs text-only without its vision encoder; it's only
            # needed for image input.
            for name, role in found.items():
                if name.startswith("vision_encoder"):
                    role.required = False
            if not any(role.has_vision_weights for role in found.values()):
                declared_by = self._vision_config_source(model_id, info, target_filenames)
                if declared_by is not None:
                    notes.append(
                        f"{declared_by}'s config declares a vision encoder, but no vision weights were found "
                        "in this repo - image input needs a matching mmproj/vision encoder from elsewhere"
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
    def _find_same_model_role(
        found: dict[str, HfRequirementRole], role_name: str, architecture: str | None
    ) -> HfRequirementRole | None:
        """The role an identified file joins: the one with its architecture or,
        when the signature names none, the same-named role that doesn't either -
        so a repo's quantizations of one LLM (or its mmproj) group together."""
        if architecture is not None:
            return next((role for role in found.values() if role.architecture == architecture), None)
        existing = found.get(role_name)
        return existing if existing is not None and existing.architecture is None else None

    @staticmethod
    def _is_text_encoder_role(role_name: str) -> bool:
        return role_name.startswith(("text_encoder", "language_model"))

    @staticmethod
    def _has_vision_hint(filename: str) -> bool:
        basename = filename.rsplit("/", 1)[-1].lower()
        return any(hint in basename for hint in VISION_FILENAME_HINTS)

    def _weight_map(self, repo_id: str, index_file: str) -> dict[str, str]:
        """Tensor name -> shard (full repo path), from a sharded checkpoint's index."""
        index_path = hf_hub_download(repo_id=repo_id, filename=index_file, token=self.token)
        with open(index_path, encoding="utf-8") as file:
            index = json.load(file)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            raise ValueError("index has no weight_map")
        folder = index_file.rpartition("/")[0]
        return {name: self._repo_path(folder, shard) for name, shard in weight_map.items()}

    def _read_weight_maps(
        self, repo_id: str, index_files: list[str], notes: list[str]
    ) -> dict[str, dict[str, str]]:
        def read(index_file: str) -> dict[str, str] | None:
            try:
                return self._weight_map(repo_id, index_file)
            except (HfHubHTTPError, OSError, ValueError) as error:
                notes.append(f"could not read shard index {index_file} ({error})")
                return None

        return {
            index_file: weight_map
            for index_file, weight_map in zip(index_files, self._run_parallel(read, index_files))
            if weight_map
        }

    def _has_vision_weights(
        self, repo_id: str, paths: list[str], headers: dict[str, TensorShapes]
    ) -> bool | None:
        """Whether a component's weight files (full repo paths) include a
        vision encoder. Uses headers already read, the shard index when there
        is one (the vision tower is often only in the last shard), and
        otherwise reads one weight file's header. None if nothing was readable."""
        key_sets = [list(headers[path]) for path in paths if path in headers]
        index_file = next((path for path in paths if path.endswith(SHARD_INDEX_SUFFIX)), None)
        weight_file = next((path for path in paths if self._extension(path) in FINGERPRINT_EXTENSIONS), None)
        try:
            if index_file is not None:
                key_sets.append(list(self._weight_map(repo_id, index_file)))
            elif not key_sets and weight_file is not None:
                key_sets.append(list(read_tensor_shapes(self.api, self.token, repo_id, weight_file)))
        except (HfHubHTTPError, OSError, ValueError):
            pass
        if not key_sets:
            return None
        return any(has_vision_tensors(keys) for keys in key_sets)

    def _mark_vision_weights(self, checks: list[_VisionCheck]) -> None:
        results = self._run_parallel(lambda check: self._has_vision_weights(*check[1:]), checks)
        for (role, _, _, _), has_vision in zip(checks, results):
            role.has_vision_weights = has_vision

    def _vision_config_source(self, model_id: str, info: Any, target_filenames: list[str]) -> str | None:
        """The repo itself or one of its base models whose config.json declares
        a `vision_config` (transformers' marker of a VL model), if any."""
        candidates = ([model_id] if "config.json" in target_filenames else []) + self._extract_base_model_ids(
            info
        )[:MAX_VISION_CONFIG_REPOS]
        for repo_id in candidates:
            try:
                config_path = hf_hub_download(repo_id=repo_id, filename="config.json", token=self.token)
                with open(config_path, encoding="utf-8") as file:
                    config = json.load(file)
            except (HfHubHTTPError, OSError, ValueError):
                continue
            if isinstance(config, dict) and isinstance(config.get("vision_config"), dict):
                return repo_id
        return None

    @staticmethod
    def _family_has_architecture(family: FamilyDef, architecture: str | None) -> bool:
        return architecture is not None and any(role.architecture == architecture for role in family.roles)

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
        if not isinstance(value, list):
            return False
        if len(value) == 3 and isinstance(value[2], dict):
            value = value[:2]
        return len(value) == 2 and all(item is None or isinstance(item, str) for item in value)

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
