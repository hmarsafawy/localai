"""A small, hand-maintained table of known pipeline families.

This exists for the cases nothing else can cover: a repo with no
`model_index.json`, no `base_model` relation, and no card metadata at all
(e.g. `comfyanonymous/flux_text_encoders`, which is just bare
`clip_l.safetensors` / `t5xxl_*.safetensors` files). There, the only way to
know "this is FLUX's text encoder" or to name a canonical companion repo is to
hardcode it. It necessarily needs manual upkeep as new architectures show up.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegistryRole:
    role: str
    architecture: str | None
    required: bool
    filename_aliases: tuple[str, ...]
    companion_repo: str | None = None
    companion_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class FamilyDef:
    name: str
    pipeline_classes: tuple[str, ...]
    keywords: tuple[str, ...]
    roles: tuple[RegistryRole, ...] = field(default_factory=tuple)


FAMILIES: tuple[FamilyDef, ...] = (
    FamilyDef(
        name="flux",
        pipeline_classes=(
            "FluxPipeline",
            "FluxImg2ImgPipeline",
            "FluxInpaintPipeline",
            "FluxControlNetPipeline",
            "FluxFillPipeline",
            "FluxKontextPipeline",
        ),
        keywords=("flux",),
        roles=(
            RegistryRole(
                "transformer",
                "FluxTransformer2DModel",
                True,
                ("flux1-dev", "flux1-schnell", "transformer"),
            ),
            RegistryRole(
                "text_encoder",
                "CLIPTextModel",
                True,
                ("clip_l", "clip-l", "clip_vit_l"),
                "comfyanonymous/flux_text_encoders",
                ("clip_l.safetensors",),
            ),
            RegistryRole(
                "text_encoder_2",
                "T5EncoderModel",
                True,
                ("t5xxl", "t5-xxl", "t5_xxl"),
                "comfyanonymous/flux_text_encoders",
                ("t5xxl_fp16.safetensors", "t5xxl_fp8_e4m3fn.safetensors"),
            ),
            RegistryRole(
                "vae",
                "AutoencoderKL",
                True,
                ("ae.safetensors", "vae"),
                "black-forest-labs/FLUX.1-dev",
                ("ae.safetensors",),
            ),
        ),
    ),
    FamilyDef(
        name="stable-diffusion-xl",
        pipeline_classes=(
            "StableDiffusionXLPipeline",
            "StableDiffusionXLImg2ImgPipeline",
            "StableDiffusionXLInpaintPipeline",
            "StableDiffusionXLControlNetPipeline",
        ),
        keywords=("sdxl", "stable-diffusion-xl"),
        roles=(
            RegistryRole("unet", "UNet2DConditionModel", True, ("unet",)),
            RegistryRole(
                "vae",
                "AutoencoderKL",
                True,
                ("vae", "sdxl_vae"),
                "madebyollin/sdxl-vae-fp16-fix",
                ("diffusion_pytorch_model.safetensors",),
            ),
            RegistryRole("text_encoder", "CLIPTextModel", True, ("clip_l", "clip-l", "text_encoder")),
            RegistryRole(
                "text_encoder_2", "CLIPTextModelWithProjection", True, ("clip_g", "clip-g", "text_encoder_2")
            ),
        ),
    ),
    FamilyDef(
        name="stable-diffusion-3",
        pipeline_classes=(
            "StableDiffusion3Pipeline",
            "StableDiffusion3Img2ImgPipeline",
            "StableDiffusion3InpaintPipeline",
        ),
        keywords=("stable-diffusion-3", "sd3"),
        roles=(
            RegistryRole("transformer", "SD3Transformer2DModel", True, ("sd3", "transformer")),
            RegistryRole("vae", "AutoencoderKL", True, ("vae",)),
            RegistryRole("text_encoder", "CLIPTextModel", True, ("clip_l", "clip-l")),
            RegistryRole("text_encoder_2", "CLIPTextModelWithProjection", True, ("clip_g", "clip-g")),
            RegistryRole(
                "text_encoder_3",
                "T5EncoderModel",
                False,
                ("t5xxl", "t5-xxl", "t5_xxl"),
                "stabilityai/stable-diffusion-3.5-large",
                ("text_encoder_3/model-00001-of-00002.safetensors",),
            ),
        ),
    ),
    FamilyDef(
        name="stable-diffusion",
        pipeline_classes=(
            "StableDiffusionPipeline",
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            "StableDiffusionControlNetPipeline",
        ),
        keywords=("stable-diffusion-v1", "stable-diffusion-2", "stable-diffusion"),
        roles=(
            RegistryRole("unet", "UNet2DConditionModel", True, ("unet",)),
            RegistryRole("vae", "AutoencoderKL", True, ("vae",)),
            RegistryRole("text_encoder", "CLIPTextModel", True, ("clip", "text_encoder")),
        ),
    ),
    FamilyDef(
        name="wan",
        pipeline_classes=("WanPipeline", "WanImageToVideoPipeline"),
        keywords=("wan2.1", "wan2.2", "wan-video"),
        roles=(
            RegistryRole("transformer", None, True, ("transformer",)),
            RegistryRole("vae", None, True, ("vae", "wan_vae")),
            RegistryRole("text_encoder", "UMT5EncoderModel", True, ("umt5", "t5")),
        ),
    ),
    FamilyDef(
        name="stable-audio",
        pipeline_classes=("StableAudioPipeline",),
        keywords=("stable-audio",),
        roles=(
            RegistryRole("transformer", "StableAudioDiTModel", True, ("transformer",)),
            RegistryRole("vae", "AutoencoderOobleck", True, ("autoencoder", "vae")),
            RegistryRole("text_encoder", "T5EncoderModel", True, ("t5",)),
        ),
    ),
)


def match_family_by_class(pipeline_class: str | None) -> FamilyDef | None:
    if not pipeline_class:
        return None
    for family in FAMILIES:
        if pipeline_class in family.pipeline_classes:
            return family
    return None


def match_family_by_keywords(*texts: str | None) -> FamilyDef | None:
    haystack = " ".join(text.lower() for text in texts if text)
    if not haystack:
        return None
    for family in FAMILIES:
        if any(keyword in haystack for keyword in family.keywords):
            return family
    return None


def match_filename_alias(filename: str) -> tuple[FamilyDef, RegistryRole] | None:
    """Global search across every family - only meant for discovering the family
    itself from a bare file with no other signal at all (e.g. a companion repo
    with no base_model tag, no config.json). Once a family is already known,
    use `match_filename_alias_in_family` instead so an alias belonging to a
    different family can't hijack a role (aliases are not required to be
    globally unique, only unique enough within their own family)."""
    stem = filename.lower()
    for family in FAMILIES:
        for role in family.roles:
            if any(alias in stem for alias in role.filename_aliases):
                return family, role
    return None


def match_filename_alias_in_family(filename: str, family: FamilyDef) -> RegistryRole | None:
    stem = filename.lower()
    for role in family.roles:
        if any(alias in stem for alias in role.filename_aliases):
            return role
    return None
