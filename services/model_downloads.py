"""Downloads the quantization and vision encoder someone chose from a Hugging Face
text model, and keeps track of what's on disk.

Each repo gets a folder holding its files under their repo paths, plus a manifest.
A file only gets its real name once its checksum has passed, so what's complete is
worked out from the disk rather than trusted from a record of it.

A repo with no GGUF files is converted instead: its safetensors are downloaded
into a hidden folder, converted to GGUF by llama.cpp's converter for the active
runtime, and quantized. Only the GGUF is kept.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import HfHubHTTPError, HFValidationError

from models.downloads import LocalModel, LocalModelOption, ModelLibrary
from models.hf_model import HfModelRequirements, HfRequirementRole, HfRoleOption
from models.jobs import JobInfo
from services.hardware import hardware_service
from services.huggingface import HuggingFaceService
from services.jobs import ACTIVE_STATUSES, JobContext, JobStore, download_file
from services.llama_converter import LlamaConverterService, llama_converter_service
from services.llama_runtime import LaunchableRuntime, RuntimeConflictError, RuntimeNotFoundError, parse_build_name
from services.model_conversion import (
    CONVERTER_OUTTYPES,
    INTERMEDIATE,
    conversion_options,
    convert,
    estimated_size_bytes,
    gguf_name,
    quantize,
)
from services.model_options import (
    MemoryBudget,
    OptionChoiceError,
    annotate_fit,
    default_option,
    language_model_options,
    match_option,
    vision_encoder_options,
)
from services.settings import TEXT_GENERATION_TASK, text_models_dir

logger = logging.getLogger(__name__)

MANIFEST_NAME = "localai-model.json"
PARTIAL_SUFFIX = ".partial"
LANGUAGE_MODEL = "language_model"
VISION_ENCODER = "vision_encoder"
AUTO = "auto"
NONE = "none"

# A drive left completely full misbehaves in ways that have nothing to do with us.
SPACE_MARGIN_BYTES = 1_000_000_000
# A drop that made headway costs nothing to resume, so only fruitless tries count.
MAX_ATTEMPTS_WITHOUT_PROGRESS = 5
RETRY_DELAY_SECONDS = 2.0
RETRYABLE_STATUSES = (408, 429)

HUB_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

# Where a conversion's download and its BF16 go, inside the model's folder.
SOURCE_DIR = ".source"
SOURCE_RECORD = "localai-source.json"
# What the converter reads besides the weights: the config, the tokenizer and
# the chat template. Never code: the converter loads tokenizers in a way that
# would run a repo's own.
SOURCE_EXTENSIONS = (".json", ".model", ".txt", ".jinja", ".tiktoken")


class ModelDownloadRequestError(Exception):
    """The repo or the choice can't be downloaded, and trying again won't change that."""


class ModelAccessError(Exception):
    """The repo is listed publicly, but its files need a license accepted first."""


class HubUnavailableError(Exception):
    """Hugging Face couldn't be reached."""


class NotEnoughSpaceError(Exception):
    pass


class LocalModelNotFoundError(Exception):
    pass


class LocalModelBusyError(Exception):
    """It's downloading, or something has its files open."""


class LocalModelChoiceError(Exception):
    """A quantization or vision encoder that doesn't pick out one downloaded option."""


class ModelDownloadError(Exception):
    """A download that started and couldn't finish."""


@dataclass(frozen=True)
class _RepoFile:
    path: str
    size: int
    sha256: str | None


@dataclass(frozen=True)
class _Choice:
    role: str
    option: HfRoleOption
    files: tuple[_RepoFile, ...]

    @property
    def size_bytes(self) -> int:
        return sum(file.size for file in self.files)


@dataclass(frozen=True)
class LocalSelection:
    """Downloaded files to run: a quantization, and a vision encoder if wanted."""

    repo_id: str
    model: LocalModelOption
    vision: LocalModelOption | None


@dataclass(frozen=True)
class _ConversionPlan:
    """A safetensors repo to convert, worked out before the job starts."""

    repo_id: str
    revision: str | None
    architecture: str
    parameters: int
    files: tuple[_RepoFile, ...]
    target: HfRoleOption
    runtime: LaunchableRuntime
    tag: str
    notes: tuple[str, ...]

    @property
    def source_bytes(self) -> int:
        return sum(file.size for file in self.files)

    @property
    def direct(self) -> bool:
        """Whether the converter writes the target itself, without `llama-quantize`."""
        return self.target.id in CONVERTER_OUTTYPES

    @property
    def disk_bytes(self) -> int:
        """The most it takes at once. The download goes once it's converted, so
        that's the download and the BF16, or the BF16 and the quantization."""
        if self.direct:
            return self.source_bytes + (self.target.size_bytes or 0)
        intermediate = estimated_size_bytes(self.parameters, INTERMEDIATE)
        return max(self.source_bytes + intermediate, intermediate + (self.target.size_bytes or 0))


# (repo id, the (role, option id) pairs being deleted or None for all) -> why not.
BusyCheck = Callable[[str, set[tuple[str, str]] | None], str | None]


def repo_folder_name(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def _is_safe_repo_path(path: str) -> bool:
    """Filenames come from whoever owns the repo and end up as paths on this
    machine, so one mustn't be able to reach outside the repo's folder. Windows
    ignores trailing dots and spaces, which makes `.. ` another way to say `..`."""
    if any(character in "\\:" or ord(character) < 32 for character in path):
        return False
    posix = PurePosixPath(path)
    return bool(posix.parts) and not posix.is_absolute() and all(part.rstrip(". ") for part in posix.parts)


def _gguf_files(info: Any) -> dict[str, _RepoFile]:
    files: dict[str, _RepoFile] = {}
    for sibling in getattr(info, "siblings", None) or []:
        if sibling.rfilename.lower().endswith(".gguf"):
            lfs = getattr(sibling, "lfs", None)
            files[sibling.rfilename] = _RepoFile(
                sibling.rfilename, getattr(sibling, "size", None) or 0, getattr(lfs, "sha256", None)
            )
    return files


def _source_files(repo_id: str, info: Any, weights: HfRoleOption) -> tuple[_RepoFile, ...]:
    """The weights, and what the converter reads next to them in the repo's
    top folder."""
    if any("/" in path for path in weights.files):
        raise ModelDownloadRequestError(
            f"{repo_id} keeps its weights in a subfolder, which converting doesn't support."
        )
    wanted = set(weights.files)
    files: list[_RepoFile] = []
    for sibling in getattr(info, "siblings", None) or []:
        path = sibling.rfilename
        if path in wanted or ("/" not in path and path.lower().endswith(SOURCE_EXTENSIONS)):
            if not _is_safe_repo_path(path):
                raise ModelDownloadRequestError(f"Refusing to download a file with an unsafe path: {path!r}")
            lfs = getattr(sibling, "lfs", None)
            files.append(_RepoFile(path, getattr(sibling, "size", None) or 0, getattr(lfs, "sha256", None)))
    missing = wanted - {file.path for file in files}
    if missing:
        raise ModelDownloadRequestError(f"{', '.join(sorted(missing))} isn't in the repo any more. Please try again.")
    return tuple(files)


def _architecture(info: Any) -> str | None:
    """What the model is, as its `config.json` names it (`LlamaForCausalLM`)."""
    config = getattr(info, "config", None) or {}
    architectures = config.get("architectures") if isinstance(config, dict) else None
    if isinstance(architectures, list) and architectures and isinstance(architectures[0], str):
        return architectures[0]
    return None


def _parameter_count(info: Any) -> int | None:
    """From the Hub's reading of the safetensors headers, when it has one."""
    total = getattr(getattr(info, "safetensors", None), "total", None)
    return total if isinstance(total, int) and total > 0 else None


def _source_revision(source: Path) -> str | None:
    """The repo version a conversion's download is of."""
    try:
        revision = json.loads((source / SOURCE_RECORD).read_text(encoding="utf-8")).get("revision")
    except (OSError, ValueError, AttributeError):
        return None
    return revision if isinstance(revision, str) else None


def _tree_size(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(_size_of(path) for path in folder.rglob("*") if path.is_file())


def _partial_path(path: Path) -> Path:
    return path.with_name(path.name + PARTIAL_SUFFIX)


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _free_bytes(path: Path) -> int | None:
    """Free space where `path` is or would be, since it may not exist yet."""
    for candidate in (path, *path.parents):
        if candidate.exists():
            try:
                return shutil.disk_usage(candidate).free
            except OSError:
                return None
    return None


def _read_manifest(folder: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        logger.warning("Ignoring the unreadable manifest in %s: %s", folder, error)
        return None
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("options"), list)
        or not isinstance(manifest.get("files"), dict)
    ):
        logger.warning("Ignoring the malformed manifest in %s", folder)
        return None
    return manifest


def _write_manifest(folder: Path, manifest: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, folder / MANIFEST_NAME)


def _is_complete(folder: Path, path: str, entry: dict[str, Any] | None) -> bool:
    if not entry or not entry.get("verified"):
        return False
    final = folder / path
    return final.is_file() and _size_of(final) == entry.get("size")


class ModelDownloadService:
    def __init__(
        self,
        hf: HuggingFaceService | None = None,
        budget: Callable[[], MemoryBudget | None] | None = None,
        client: httpx.Client | None = None,
        jobs: JobStore | None = None,
        root: Path | None = None,
        converter: LlamaConverterService | None = None,
    ) -> None:
        self.hf = hf or HuggingFaceService()
        self._budget = budget or hardware_service.memory_budget
        self.converter = converter or llama_converter_service
        token = self.hf.token
        # httpx drops this header when a download redirects to another host, so
        # the token only ever reaches Hugging Face and not its CDN.
        self.client = client or httpx.Client(
            timeout=HUB_TIMEOUT, headers={"Authorization": f"Bearer {token}"} if token else None
        )
        # One at a time: a second download would only share the same connection,
        # and two jobs could otherwise write one vision encoder file at once.
        self.jobs = jobs or JobStore(max_concurrent=1)
        self.root = root or text_models_dir()
        self._lock = threading.Lock()
        self._busy_checks: list[BusyCheck] = []

    def add_busy_check(self, check: BusyCheck) -> None:
        """`check(repo_id, options)` returns why those files can't be deleted
        right now, or None. `options` holds the (role, option id) pairs being
        deleted, or is None for the whole repo. It's how a running model keeps
        its own files without blocking the deletion of others."""
        self._busy_checks.append(check)

    # Starting a download

    def start(self, repo_id: str, quantization: str = AUTO, vision_encoder: str = AUTO) -> JobInfo:
        """Looks everything up first, so a bad repo or choice is an immediate
        error rather than a failed job. Then the work runs in the background.
        A repo without GGUF files is converted from its safetensors."""
        conversion: _ConversionPlan | None = None
        try:
            info = self.hf.api.model_info(repo_id, files_metadata=True)
            if _gguf_files(info):
                resolved_id, revision, model, vision = self._resolve(info, repo_id, quantization, vision_encoder)
            else:
                conversion = self._plan_conversion(info, repo_id, quantization, vision_encoder)
        except HfHubHTTPError:
            raise  # Raised over the network, but the Hub did answer.
        except HFValidationError as error:
            raise ModelDownloadRequestError(f"'{repo_id}' isn't a valid Hugging Face repo: {error}") from error
        except (OSError, httpx.TransportError) as error:
            raise HubUnavailableError(f"Could not reach Hugging Face: {error}") from error
        if conversion is not None:
            return self._start_conversion(conversion)

        choices = [model, vision] if vision else [model]
        self._check_space(self._folder(resolved_id), choices)

        warning = None
        if quantization.casefold() != AUTO and model.option.fit == "too_large":
            warning = (
                f"{model.option.id} is estimated not to fit in this machine's free memory, "
                "so it may not run. It's being downloaded anyway, as you chose it."
            )

        def work(context: JobContext) -> dict[str, Any]:
            if warning:
                context.note(warning)
            return self._download(context, resolved_id, revision, model, vision)

        vision_id = vision.option.id if vision else None
        return self.jobs.submit(
            "model_download",
            f"{resolved_id}:{model.option.id}+{vision_id or NONE}",
            work,
            details={
                "repo_id": resolved_id,
                "task": TEXT_GENERATION_TASK,
                "revision": revision,
                "quantization": model.option.id,
                "vision_encoder": vision_id,
                "size_bytes": sum(choice.size_bytes for choice in choices),
            },
        )

    def _resolve(
        self, info: Any, repo_id: str, quantization: str, vision_encoder: str
    ) -> tuple[str, str | None, _Choice, _Choice | None]:
        resolved_id = getattr(info, "id", None) or repo_id
        revision = getattr(info, "sha", None)
        files = _gguf_files(info)
        # A gated repo's listing is public and only its files are refused, and
        # without being able to read them it wouldn't be recognised as anything.
        self._check_access(resolved_id, revision, next(iter(files)))

        requirements = self.hf.get_model_requirements(resolved_id)
        annotate_fit(requirements, self._budget())

        model_options = language_model_options(requirements)
        if not model_options:
            raise ModelDownloadRequestError(self._not_a_text_model(resolved_id, requirements))
        model = self._choice(LANGUAGE_MODEL, self._pick_model(model_options, quantization), files)
        vision_option = self._pick_vision(vision_encoder_options(requirements), vision_encoder, resolved_id)
        vision = self._choice(VISION_ENCODER, vision_option, files) if vision_option else None
        return resolved_id, revision, model, vision

    def _check_access(self, repo_id: str, revision: str | None, path: str) -> None:
        response = self.client.head(hf_hub_url(repo_id, path, revision=revision), follow_redirects=False)
        if response.status_code in (401, 403):
            raise ModelAccessError(
                f"{repo_id} is gated. Accept its license at https://huggingface.co/{repo_id}, "
                "then make sure HF_TOKEN in .env is a token from that account."
            )

    @staticmethod
    def _not_a_text_model(repo_id: str, requirements: HfModelRequirements) -> str:
        found = [role.role for role in requirements.roles if role.present]
        if found:
            return f"{repo_id} isn't a text model. It has these parts: {', '.join(found)}."
        return f"{repo_id} couldn't be recognised as a text model."

    @staticmethod
    def _pick_model(options: list[HfRoleOption], wanted: str) -> HfRoleOption:
        if wanted.casefold() == AUTO:
            chosen = default_option(options)
            if chosen is None:
                raise ModelDownloadRequestError(
                    "None of this repo's quantizations look like they'd fit on this machine, so pick one "
                    f"yourself. Available: {', '.join(option.id for option in options)}"
                )
            return chosen
        try:
            return match_option(options, wanted)
        except OptionChoiceError as error:
            raise ModelDownloadRequestError(f"Quantization {error}") from error

    @staticmethod
    def _pick_vision(options: list[HfRoleOption], wanted: str, repo_id: str) -> HfRoleOption | None:
        folded = wanted.casefold()
        if folded == NONE:
            return None
        if folded == AUTO:
            return default_option(options) or (options[0] if options else None)
        if not options:
            raise ModelDownloadRequestError(f"{repo_id} has no vision encoder to download.")
        try:
            return match_option(options, wanted)
        except OptionChoiceError as error:
            raise ModelDownloadRequestError(f"Vision encoder {error}") from error

    @staticmethod
    def _choice(role: str, option: HfRoleOption, files: dict[str, _RepoFile]) -> _Choice:
        chosen: list[_RepoFile] = []
        # Only GGUF: it's all llama.cpp reads, and it keeps a repo's `.py` files
        # from landing in a folder that restarts the server when one appears.
        for path in option.files:
            if not path.lower().endswith(".gguf"):
                continue
            if not _is_safe_repo_path(path):
                raise ModelDownloadRequestError(f"Refusing to download a file with an unsafe path: {path!r}")
            if path not in files:
                raise ModelDownloadRequestError(f"{path} isn't in the repo any more. Please try again.")
            chosen.append(files[path])
        if not chosen:
            raise ModelDownloadRequestError(f"{option.id} has no GGUF files.")
        return _Choice(role, option, tuple(chosen))

    def _folder(self, repo_id: str) -> Path:
        return self.root / repo_folder_name(repo_id)

    def _check_space(self, folder: Path, choices: list[_Choice]) -> None:
        """Counts what's still to fetch, so a nearly finished download can resume
        on a drive that wouldn't fit the whole file again."""
        manifest = _read_manifest(folder) or {"files": {}}
        files = {file.path: file for choice in choices for file in choice.files}
        needed = 0
        for path, file in files.items():
            entry = manifest["files"].get(path)
            if not self._up_to_date(folder, file, entry):
                needed += max(file.size - self._usable_partial_bytes(folder, file, entry), 0)
        if not needed:
            return
        free = _free_bytes(self.root)
        if free is not None and free < needed + SPACE_MARGIN_BYTES:
            raise NotEnoughSpaceError(
                f"Not enough free space on the drive holding {self.root}: this needs about "
                f"{needed / 1e9:.1f} GB and {free / 1e9:.1f} GB is free. "
                "Free some up, or set LOCALAI_MODELS_DIR to a bigger drive."
            )

    @staticmethod
    def _same_version(file: _RepoFile, entry: dict[str, Any] | None) -> bool:
        return bool(entry) and entry.get("sha256") == file.sha256 and entry.get("size") == file.size

    @classmethod
    def _up_to_date(cls, folder: Path, file: _RepoFile, entry: dict[str, Any] | None) -> bool:
        return cls._same_version(file, entry) and _is_complete(folder, file.path, entry)

    @classmethod
    def _usable_partial_bytes(cls, folder: Path, file: _RepoFile, entry: dict[str, Any] | None) -> int:
        """A partial of an older version of the file is worthless. With no record
        of its version, it's still worth trying: the checksum will catch it."""
        if entry and not cls._same_version(file, entry):
            return 0
        return _size_of(_partial_path(folder / file.path))

    # Starting a conversion

    def _plan_conversion(self, info: Any, repo_id: str, quantization: str, vision_encoder: str) -> _ConversionPlan:
        resolved_id = getattr(info, "id", None) or repo_id
        revision = getattr(info, "sha", None)
        requirements = self.hf.get_model_requirements(resolved_id)
        weights = self._safetensors_weights(resolved_id, requirements)
        files = _source_files(resolved_id, info, weights)
        self._check_access(resolved_id, revision, weights.files[0])
        if vision_encoder.casefold() not in (AUTO, NONE):
            raise ModelDownloadRequestError(
                f"{resolved_id} has no GGUF vision encoder, and converting one isn't supported yet. "
                "Leave vision_encoder as auto or none."
            )
        architecture = _architecture(info)
        if architecture is None:
            raise ModelDownloadRequestError(
                f"{resolved_id} has no GGUF files, and its config.json doesn't say what model it is, "
                "so it can't be converted."
            )

        try:
            runtime = self.converter.runtimes.launchable()
        except (RuntimeNotFoundError, RuntimeConflictError) as error:
            raise ModelDownloadRequestError(
                f"{resolved_id} has no GGUF files, so it has to be converted, which needs a llama.cpp runtime: {error}"
            ) from error
        build = parse_build_name(runtime.id)
        tag = build.tag if build else runtime.id
        supported = self.converter.architectures(tag)
        if supported is not None and architecture not in supported:
            raise ModelDownloadRequestError(self._unconvertible(resolved_id, architecture, tag))

        parameters = _parameter_count(info) or (weights.size_bytes or 0) // 2
        options = conversion_options(resolved_id, parameters)
        annotate_fit(
            HfModelRequirements(
                model_id=resolved_id,
                roles=[HfRequirementRole(role=LANGUAGE_MODEL, present=True, options=options)],
            ),
            self._budget(),
        )
        target = self._pick_model(options, quantization)

        notes: list[str] = []
        if supported is None:
            notes.append(
                f"llama.cpp's converter for {tag} isn't set up yet, so it's set up first. "
                "That's needed once, and takes about 1 GB."
            )
        if any(role.has_vision_weights for role in requirements.roles) or any(
            role.role.startswith(VISION_ENCODER) and role.present for role in requirements.roles
        ):
            notes.append("Converted as a text model: converting its vision encoder isn't supported yet.")
        if quantization.casefold() != AUTO and target.fit == "too_large":
            notes.append(
                f"{target.id} is estimated not to fit in this machine's free memory, "
                "so it may not run. It's being converted anyway, as you chose it."
            )
        return _ConversionPlan(
            repo_id=resolved_id,
            revision=revision,
            architecture=architecture,
            parameters=parameters,
            files=files,
            target=target,
            runtime=runtime,
            tag=tag,
            notes=tuple(notes),
        )

    @staticmethod
    def _safetensors_weights(repo_id: str, requirements: HfModelRequirements) -> HfRoleOption:
        options = [
            option
            for role in requirements.roles
            if role.role.startswith(LANGUAGE_MODEL)
            for option in role.options
            if option.format == "safetensors"
        ]
        if not options:
            formats = sorted({option.format for role in requirements.roles for option in role.options if option.format})
            found = f" It has {', '.join(formats)} weights, which aren't safe or can't be converted." if formats else ""
            raise ModelDownloadRequestError(
                f"{repo_id} has neither GGUF files to run nor safetensors weights to convert.{found}"
            )
        # The plain weights, rather than a variant such as an int8 copy.
        plain = [option for option in options if option.variant is None and option.quantization is None]
        return min(plain or options, key=lambda option: option.size_bytes or 0)

    @staticmethod
    def _unconvertible(repo_id: str, architecture: str, tag: str) -> str:
        return (
            f"llama.cpp {tag} can't convert {architecture} models, so {repo_id} can't be converted. "
            "Look for a GGUF version of it on Hugging Face, or try a newer llama.cpp runtime."
        )

    def _start_conversion(self, plan: _ConversionPlan) -> JobInfo:
        self._check_conversion_space(plan)

        def work(context: JobContext) -> dict[str, Any]:
            for note in plan.notes:
                context.note(note)
            return self._convert(context, plan)

        return self.jobs.submit(
            "model_conversion",
            f"{plan.repo_id}:{plan.target.id}+{NONE}",
            work,
            details={
                "repo_id": plan.repo_id,
                "task": TEXT_GENERATION_TASK,
                "revision": plan.revision,
                "quantization": plan.target.id,
                "vision_encoder": None,
                "size_bytes": plan.source_bytes,
                "conversion": {
                    "source_format": "safetensors",
                    "architecture": plan.architecture,
                    "parameters": plan.parameters,
                    "converter": plan.tag,
                    "estimated_size_bytes": plan.target.size_bytes,
                    "disk_needed_bytes": plan.disk_bytes,
                },
            },
        )

    def _check_conversion_space(self, plan: _ConversionPlan) -> None:
        """What's downloaded or converted already counts towards the peak."""
        folder = self._folder(plan.repo_id)
        needed = max(plan.disk_bytes - _tree_size(folder / SOURCE_DIR), 0)
        free = _free_bytes(self.root)
        if free is not None and free < needed + SPACE_MARGIN_BYTES:
            raise NotEnoughSpaceError(
                f"Not enough free space on the drive holding {self.root}: converting {plan.repo_id} to "
                f"{plan.target.id} needs about {needed / 1e9:.1f} GB at once, and {free / 1e9:.1f} GB is free. "
                "Free some up, or set LOCALAI_MODELS_DIR to a bigger drive."
            )

    # The background work

    def _download(
        self,
        context: JobContext,
        repo_id: str,
        revision: str | None,
        model: _Choice,
        vision: _Choice | None,
    ) -> dict[str, Any]:
        folder = self._folder(repo_id)
        choices = [model, vision] if vision else [model]
        # It may have sat in the queue while another download used the space up.
        self._check_space(folder, choices)
        self._record_choices(repo_id, revision, choices)

        files = list({file.path: file for choice in choices for file in choice.files}.values())
        total = sum(file.size for file in files)
        manifest = _read_manifest(folder) or {"files": {}}
        pending = [file for file in files if not _is_complete(folder, file.path, manifest["files"].get(file.path))]
        pending_paths = {file.path for file in pending}
        context.update(stage="downloading")
        context.begin_transfer(
            total,
            sum(
                _size_of(_partial_path(folder / file.path)) if file.path in pending_paths else file.size
                for file in files
            ),
        )

        base_bytes = 0
        for file in files:
            if file.path in pending_paths:
                self._fetch(context, repo_id, revision, folder, file, base_bytes)
            base_bytes += file.size

        context.update(stage=None, current_file=None)
        context.progress(total)
        return {
            "repo_id": repo_id,
            "task": TEXT_GENERATION_TASK,
            "directory": str(folder),
            "quantization": model.option.id,
            "model_path": str(folder / model.files[0].path),
            "vision_encoder": vision.option.id if vision else None,
            "vision_encoder_path": str(folder / vision.files[0].path) if vision else None,
            "already_downloaded": not pending,
        }

    def _record_choices(self, repo_id: str, revision: str | None, choices: list[_Choice]) -> None:
        """Written before any data arrives, so an unfinished download shows up in
        the library and can be deleted from there."""
        folder = self._folder(repo_id)
        with self._lock:
            manifest = _read_manifest(folder) or {
                "repo_id": repo_id,
                "task": TEXT_GENERATION_TASK,
                "options": [],
                "files": {},
            }
            for choice in choices:
                self._record_option(manifest, choice)
                for file in choice.files:
                    entry = manifest["files"].get(file.path)
                    if entry and not self._same_version(file, entry):
                        # The repo changed. Any partial is of the old version, but a
                        # finished file stays usable until its replacement is verified.
                        _partial_path(folder / file.path).unlink(missing_ok=True)
                        entry = None
                    manifest["files"][file.path] = {
                        "size": file.size,
                        "sha256": file.sha256,
                        "revision": entry["revision"] if entry and entry.get("verified") else revision,
                        "verified": bool(entry and entry.get("verified")),
                    }
            _write_manifest(folder, manifest)

    @staticmethod
    def _record_option(manifest: dict[str, Any], choice: _Choice) -> None:
        record = {
            "role": choice.role,
            "id": choice.option.id,
            "label": choice.option.label,
            "quantization": choice.option.quantization,
            "variant": choice.option.variant,
            "files": [file.path for file in choice.files],
            "shards": len(choice.files),
            "size_bytes": choice.size_bytes,
        }
        for index, existing in enumerate(manifest["options"]):
            if existing.get("role") == choice.role and existing.get("id") == choice.option.id:
                manifest["options"][index] = record
                return
        manifest["options"].append(record)

    def _fetch(
        self,
        context: JobContext,
        repo_id: str,
        revision: str | None,
        folder: Path,
        file: _RepoFile,
        base_bytes: int,
    ) -> None:
        final = folder / file.path
        partial = _partial_path(final)
        url = hf_hub_url(repo_id, file.path, revision=revision)
        context.update(stage="downloading", current_file=file.path)

        fruitless_attempts = 0
        noted_a_drop = False
        while True:
            before = _size_of(partial)
            try:
                download_file(
                    self.client,
                    url,
                    partial,
                    context,
                    sha256=file.sha256,
                    base_bytes=base_bytes,
                    size=file.size or None,
                )
                break
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if status in (401, 403):
                    raise ModelDownloadError(
                        f"Hugging Face refused access to {file.path} ({status}). If the repo is gated, accept its "
                        f"license at https://huggingface.co/{repo_id} and check HF_TOKEN in .env."
                    ) from error
                if status not in RETRYABLE_STATUSES and status < 500:
                    raise ModelDownloadError(
                        f"Hugging Face answered {status} for {file.path}. The repo may have changed; try again."
                    ) from error
                failure: Exception = error
            except httpx.TransportError as error:
                failure = error

            fruitless_attempts = 0 if _size_of(partial) > before else fruitless_attempts + 1
            if fruitless_attempts >= MAX_ATTEMPTS_WITHOUT_PROGRESS:
                raise ModelDownloadError(
                    f"Gave up on {file.path} after {fruitless_attempts} tries in a row with no new data "
                    f"({failure}). What's downloaded so far is kept, so starting it again will resume."
                ) from failure
            if not noted_a_drop:
                context.note(f"The connection dropped during {file.path}; resuming automatically.")
                noted_a_drop = True
            context.wait(RETRY_DELAY_SECONDS * max(fruitless_attempts, 1))

        with self._lock:
            os.replace(partial, final)
            manifest = _read_manifest(folder)
            if manifest is not None and file.path in manifest["files"]:
                manifest["files"][file.path].update(verified=True, revision=revision)
                _write_manifest(folder, manifest)

    def _convert(self, context: JobContext, plan: _ConversionPlan) -> dict[str, Any]:
        folder = self._folder(plan.repo_id)
        source = folder / SOURCE_DIR
        final = folder / plan.target.files[0]
        manifest = _read_manifest(folder)
        if manifest is not None and _is_complete(folder, final.name, manifest["files"].get(final.name)):
            return self._conversion_result(plan, final, already_converted=True)

        if self.converter.architectures(plan.tag) is None:
            self.converter.ensure_installed(context, plan.tag)
            if plan.architecture not in (self.converter.architectures(plan.tag) or []):
                raise ModelDownloadError(self._unconvertible(plan.repo_id, plan.architecture, plan.tag))
        # It may have sat in the queue while another download used the space up.
        self._check_conversion_space(plan)
        self._record_conversion(plan)

        intermediate = source / gguf_name(plan.repo_id, INTERMEDIATE)
        if plan.direct or not intermediate.is_file() or _source_revision(source) != plan.revision:
            self._download_source(context, plan, source)
            context.update(stage="converting", current_file=None)
            converted = final if plan.direct else intermediate
            partial = _partial_path(converted)
            convert(
                context,
                self.converter.command(plan.tag),
                source,
                partial,
                plan.target.id if plan.direct else INTERMEDIATE,
            )
            os.replace(partial, converted)
            # The BF16 is all that quantizing needs. Kept if quantizing fails, so
            # trying again starts from it.
            for file in plan.files:
                (source / file.path).unlink(missing_ok=True)
        if not plan.direct:
            context.update(stage="quantizing", current_file=None)
            partial = _partial_path(final)
            quantize(context, plan.runtime, intermediate, partial, plan.target.id)
            os.replace(partial, final)

        context.update(stage="finalizing", current_file=None)
        self._finish_conversion(plan, final)
        shutil.rmtree(source, ignore_errors=True)
        return self._conversion_result(plan, final, already_converted=False)

    def _download_source(self, context: JobContext, plan: _ConversionPlan, source: Path) -> None:
        """Into `source`, resuming what an earlier try left, unless that was of
        another version of the repo."""
        if _source_revision(source) != plan.revision:
            shutil.rmtree(source, ignore_errors=True)
        source.mkdir(parents=True, exist_ok=True)
        (source / SOURCE_RECORD).write_text(json.dumps({"revision": plan.revision}), encoding="utf-8")

        pending = {file.path for file in plan.files if _size_of(source / file.path) != file.size}
        context.update(stage="downloading")
        context.begin_transfer(
            plan.source_bytes,
            sum(
                _size_of(_partial_path(source / file.path)) if file.path in pending else file.size
                for file in plan.files
            ),
        )
        base_bytes = 0
        for file in plan.files:
            if file.path in pending:
                self._fetch(context, plan.repo_id, plan.revision, source, file, base_bytes)
            base_bytes += file.size
        context.update(current_file=None)
        context.progress(plan.source_bytes)

    def _record_conversion(self, plan: _ConversionPlan) -> None:
        """Written before anything is downloaded, so an unfinished conversion
        shows up in the library and can be deleted from there."""
        folder = self._folder(plan.repo_id)
        name = plan.target.files[0]
        with self._lock:
            manifest = _read_manifest(folder) or {
                "repo_id": plan.repo_id,
                "task": TEXT_GENERATION_TASK,
                "options": [],
                "files": {},
            }
            record = {
                "role": LANGUAGE_MODEL,
                "id": plan.target.id,
                "label": plan.target.label,
                "quantization": plan.target.quantization,
                "variant": None,
                "files": [name],
                "shards": 1,
                "size_bytes": plan.target.size_bytes,
                "converted": {"from_revision": plan.revision, "converter": plan.tag},
            }
            options = manifest["options"]
            matching = [
                index
                for index, existing in enumerate(options)
                if existing.get("role") == LANGUAGE_MODEL and existing.get("id") == plan.target.id
            ]
            if matching:
                options[matching[0]] = record
            else:
                options.append(record)
            entry = manifest["files"].get(name)
            if not (entry and entry.get("verified")):
                manifest["files"][name] = {"size": None, "sha256": None, "revision": plan.revision, "verified": False}
            _write_manifest(folder, manifest)

    def _finish_conversion(self, plan: _ConversionPlan, final: Path) -> None:
        folder = self._folder(plan.repo_id)
        size = _size_of(final)
        with self._lock:
            manifest = _read_manifest(folder)
            if manifest is None:
                raise ModelDownloadError(f"{plan.repo_id}'s record of its files went missing while it was converted")
            manifest["files"][final.name] = {"size": size, "sha256": None, "revision": plan.revision, "verified": True}
            for record in manifest["options"]:
                if record.get("role") == LANGUAGE_MODEL and record.get("id") == plan.target.id:
                    record["size_bytes"] = size
            _write_manifest(folder, manifest)

    def _conversion_result(self, plan: _ConversionPlan, final: Path, already_converted: bool) -> dict[str, Any]:
        return {
            "repo_id": plan.repo_id,
            "task": TEXT_GENERATION_TASK,
            "directory": str(final.parent),
            "quantization": plan.target.id,
            "model_path": str(final),
            "vision_encoder": None,
            "vision_encoder_path": None,
            "already_downloaded": already_converted,
            "converted_with": plan.tag,
            "size_bytes": _size_of(final),
        }

    # What's on disk

    def library(self) -> ModelLibrary:
        models: list[LocalModel] = []
        if self.root.is_dir():
            for folder in sorted(self.root.iterdir(), key=lambda path: path.name.casefold()):
                manifest = _read_manifest(folder) if folder.is_dir() else None
                if manifest is not None:
                    models.append(self._local_model(folder, manifest))
        return ModelLibrary(
            models_dir=str(self.root),
            free_bytes=_free_bytes(self.root),
            size_on_disk_bytes=sum(model.size_on_disk_bytes for model in models),
            models=models,
        )

    def _local_model(self, folder: Path, manifest: dict[str, Any]) -> LocalModel:
        size_on_disk = sum(
            _size_of(folder / path) + _size_of(_partial_path(folder / path)) for path in manifest["files"]
        ) + _tree_size(folder / SOURCE_DIR)
        options = sorted(
            (self._local_option(folder, manifest, record) for record in manifest["options"]),
            key=lambda option: (option[1].size_bytes, option[1].id),
        )
        return LocalModel(
            repo_id=manifest.get("repo_id") or folder.name.replace("--", "/", 1),
            task=manifest.get("task") or TEXT_GENERATION_TASK,
            directory=str(folder),
            size_on_disk_bytes=size_on_disk,
            quantizations=[option for role, option in options if role == LANGUAGE_MODEL],
            vision_encoders=[option for role, option in options if role == VISION_ENCODER],
        )

    @staticmethod
    def _local_option(
        folder: Path, manifest: dict[str, Any], record: dict[str, Any]
    ) -> tuple[str, LocalModelOption]:
        files = record.get("files") or []
        on_disk = 0
        complete = bool(files)
        for path in files:
            entry = manifest["files"].get(path)
            if _is_complete(folder, path, entry):
                on_disk += entry["size"]
            else:
                complete = False
                on_disk += _size_of(_partial_path(folder / path))
        return record.get("role"), LocalModelOption(
            id=record["id"],
            label=record.get("label") or record["id"],
            quantization=record.get("quantization"),
            variant=record.get("variant"),
            status="complete" if complete else "incomplete",
            path=str(folder / files[0]) if files else str(folder),
            files=files,
            shards=record.get("shards") or len(files),
            size_bytes=record.get("size_bytes") or 0,
            bytes_on_disk=on_disk,
            converted=bool(record.get("converted")),
        )

    # Choosing what to run

    def local_selection(self, repo_id: str, quantization: str = AUTO, vision_encoder: str = AUTO) -> LocalSelection:
        """The downloaded files to run, which have to be complete. `auto` means
        the only quantization downloaded, and a vision encoder if there is one."""
        model = self._find_local(repo_id)
        if model is None:
            raise LocalModelNotFoundError(f"{repo_id} isn't in the library. Download it with POST /downloads first.")
        downloading = self._downloading_options(model.repo_id)
        chosen = self._pick_local_model(model, quantization)
        self._check_runnable(model.repo_id, chosen, downloading)
        vision = self._pick_local_vision(model, vision_encoder)
        if vision is not None:
            self._check_runnable(model.repo_id, vision, downloading)
        return LocalSelection(repo_id=model.repo_id, model=chosen, vision=vision)

    def _find_local(self, repo_id: str) -> LocalModel | None:
        """Matched however it's capitalised, as it is on Hugging Face."""
        wanted = repo_folder_name(repo_id).casefold()
        if not self.root.is_dir():
            return None
        for folder in self.root.iterdir():
            if folder.is_dir() and folder.name.casefold() == wanted:
                manifest = _read_manifest(folder)
                if manifest is not None:
                    return self._local_model(folder, manifest)
        return None

    def _downloading_options(self, repo_id: str) -> set[str | None]:
        return {
            option
            for job in self.jobs.all_jobs()
            if job.status in ACTIVE_STATUSES and job.details.get("repo_id", "").casefold() == repo_id.casefold()
            for option in (job.details.get("quantization"), job.details.get("vision_encoder"))
        }

    @staticmethod
    def _match_local(options: list[LocalModelOption], wanted: str, what: str) -> LocalModelOption:
        candidates = [
            HfRoleOption(id=option.id, label=option.label, quantization=option.quantization) for option in options
        ]
        try:
            matched = match_option(candidates, wanted)
        except OptionChoiceError as error:
            raise LocalModelChoiceError(f"{what} {error}") from error
        return next(option for option in options if option.id == matched.id)

    def _pick_local_model(self, model: LocalModel, wanted: str) -> LocalModelOption:
        if wanted.casefold() != AUTO:
            return self._match_local(model.quantizations, wanted, "Quantization")
        complete = [option for option in model.quantizations if option.status == "complete"]
        if len(complete) == 1:
            return complete[0]
        if complete:
            raise LocalModelChoiceError(
                f"{model.repo_id} has several quantizations downloaded, so pick one: "
                f"{', '.join(option.id for option in complete)}"
            )
        if len(model.quantizations) == 1:
            return model.quantizations[0]  # Unfinished, which `_check_runnable` explains.
        raise LocalModelChoiceError(
            f"{model.repo_id} has no quantization fully downloaded. Finish one with POST /downloads, "
            f"or pick one: {', '.join(option.id for option in model.quantizations) or 'none'}"
        )

    def _pick_local_vision(self, model: LocalModel, wanted: str) -> LocalModelOption | None:
        folded = wanted.casefold()
        if folded == NONE:
            return None
        if folded != AUTO:
            if not model.vision_encoders:
                raise LocalModelChoiceError(f"{model.repo_id} has no vision encoder downloaded.")
            return self._match_local(model.vision_encoders, wanted, "Vision encoder")
        # Without one finished, it runs as text only rather than not at all.
        complete = [option for option in model.vision_encoders if option.status == "complete"]
        for preferred in ("F16", "BF16"):
            match = next((option for option in complete if option.quantization == preferred), None)
            if match is not None:
                return match
        return complete[0] if complete else None

    @staticmethod
    def _check_runnable(repo_id: str, option: LocalModelOption, downloading: set[str | None]) -> None:
        if option.id in downloading:
            raise LocalModelBusyError(f"{repo_id} {option.id} is still downloading.")
        if option.status != "complete":
            raise LocalModelBusyError(
                f"{repo_id} {option.id} is only partly downloaded "
                f"({option.bytes_on_disk / 1e9:.1f} of {option.size_bytes / 1e9:.1f} GB). "
                "Start its download again to finish it."
            )

    def delete(self, repo_id: str, quantization: str | None = None, vision_encoder: str | None = None) -> None:
        """Removes a repo's files, or only the options named, along with anything
        half downloaded. A running or queued download of the repo blocks this."""
        folder = self._folder(repo_id)
        with self._lock:
            if not folder.is_dir():
                raise LocalModelNotFoundError(f"{repo_id} isn't in the library")
            if any(
                job.status in ACTIVE_STATUSES and job.details.get("repo_id", "").casefold() == repo_id.casefold()
                for job in self.jobs.all_jobs()
            ):
                raise LocalModelBusyError(f"{repo_id} is downloading. Cancel that first.")
            whole = quantization is None and vision_encoder is None
            manifest, doomed = (
                (None, []) if whole else self._doomed_options(folder, repo_id, quantization, vision_encoder)
            )
            options = None if whole else {(record.get("role"), record["id"]) for record in doomed}
            for check in self._busy_checks:
                reason = check(repo_id, options)
                if reason:
                    raise LocalModelBusyError(reason)
            try:
                if manifest is None:
                    shutil.rmtree(folder)
                else:
                    self._delete_options(folder, manifest, doomed)
            except PermissionError as error:
                raise LocalModelBusyError(f"Something has {repo_id}'s files open: {error}") from error

    @staticmethod
    def _doomed_options(
        folder: Path, repo_id: str, quantization: str | None, vision_encoder: str | None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The manifest, and the records of the options named."""
        manifest = _read_manifest(folder)
        if manifest is None:
            raise LocalModelNotFoundError(f"{repo_id} has no readable record of its options")

        doomed: list[dict[str, Any]] = []
        for role, wanted in ((LANGUAGE_MODEL, quantization), (VISION_ENCODER, vision_encoder)):
            if wanted is None:
                continue
            records = [record for record in manifest["options"] if record.get("role") == role]
            options = [
                HfRoleOption(id=record["id"], label=record["id"], quantization=record.get("quantization"))
                for record in records
            ]
            try:
                matched = match_option(options, wanted)
            except OptionChoiceError as error:
                raise LocalModelNotFoundError(f"{repo_id}: {error}") from error
            doomed.append(next(record for record in records if record["id"] == matched.id))
        return manifest, doomed

    @staticmethod
    def _delete_options(folder: Path, manifest: dict[str, Any], doomed: list[dict[str, Any]]) -> None:
        if any(record.get("converted") for record in doomed):
            # What an unfinished conversion left. Nothing's converting: that blocks deleting.
            shutil.rmtree(folder / SOURCE_DIR, ignore_errors=True)
        manifest["options"] = [record for record in manifest["options"] if record not in doomed]
        if not manifest["options"]:
            shutil.rmtree(folder)
            return

        still_needed = {path for record in manifest["options"] for path in record.get("files") or []}
        for record in doomed:
            for path in record.get("files") or []:
                if path in still_needed:
                    continue
                target = folder / path
                target.unlink(missing_ok=True)
                _partial_path(target).unlink(missing_ok=True)
                manifest["files"].pop(path, None)
                # A split model can have a folder of its own (`BF16/...`).
                parent = target.parent
                while parent != folder and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
        _write_manifest(folder, manifest)


model_download_service = ModelDownloadService()
