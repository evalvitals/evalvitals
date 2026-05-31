"""Model spec registry — declarative ModelSpec entries, one per family.

A plain dict (no import-side-effect decorator).  Module paths are HINTS only;
the hf_local backend discovers the real decoder-layer ModuleList at load time
(see :mod:`evalvitals.models._discover`).  Token ids / merge sizes are read from
the live config via the attribute NAMES in ``VisionSpec`` — never baked as values
(GLM-4.5V 151363 vs GLM-4.1V 151343 is exactly why).

Paths reflect the verified post-VLM-refactor transformers layout (single
``.model``; Kimi/Llama4 have no outer ``.model``).  ``tool_calling=True`` marks
instruct/thinking checkpoints whose chat template renders tools (grants
TOOL_CALLS on the local backend; verified against the template at load time).
This module is torch-free.
"""

from __future__ import annotations

from evalvitals.core.spec import AttnSemantics, ModelSpec, ModulePaths, VisionSpec

REGISTRY: dict[str, ModelSpec] = {}


def _add(spec: ModelSpec) -> None:
    REGISTRY[spec.key] = spec


def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model spec {key!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def list_specs() -> list[str]:
    return sorted(REGISTRY)


# ----------------------------------------------------------------------
# LLMs (no vision tower / no TokenTypeMap)
# ----------------------------------------------------------------------
_add(ModelSpec(
    key="qwen2.5-7b-instruct", family="qwen2", model_type="qwen2",
    hf_repo="Qwen/Qwen2.5-7B-Instruct", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.43.0", tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.layers"),
    caveats=("matches the legacy QwenLLM default checkpoint",),
))
_add(ModelSpec(
    key="qwen3-4b", family="qwen3", model_type="qwen3",
    hf_repo="Qwen/Qwen3-4B", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.51.0",
    is_reasoning=True, tool_calling=True,
    chat_template_kwargs={"enable_thinking": False},  # fast/clean tool-calling for the smoke test
    module_paths=ModulePaths(decoder_layers="model.layers"),
    caveats=("small smoke-test checkpoint; q_norm/k_norm before RoPE; emits <think> by default",),
))
_add(ModelSpec(
    key="qwen3-8b", family="qwen3", model_type="qwen3",
    hf_repo="Qwen/Qwen3-8B", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.51.0",
    is_reasoning=True, tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.layers"),
    caveats=("q_norm/k_norm applied to Q/K before RoPE (account for it in lens)",),
))
_add(ModelSpec(
    key="qwen3-30b-a3b", family="qwen3_moe", model_type="qwen3_moe",
    hf_repo="Qwen/Qwen3-30B-A3B", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.51.0", is_moe=True, tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.layers", router="mlp.gate", experts="mlp.experts"),
    caveats=("v5 stores experts as fused stacked Parameters (not ModuleList) — detect at runtime",),
))
_add(ModelSpec(
    key="deepseek-v3", family="deepseek_v3", model_type="deepseek_v3",
    hf_repo="deepseek-ai/DeepSeek-V3", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.51.0", is_moe=True, tool_calling=True,
    attn_semantics=AttnSemantics.MLA_LATENT,
    module_paths=ModulePaths(decoder_layers="model.layers", router="mlp.gate", experts="mlp.experts"),
    caveats=(
        "MLA: eager materialises weights in decompressed latent head space, not raw token space",
        "HF impl naive; 671B multi-node — practical white-box on DeepSeek-V2-Lite",
    ),
))
_add(ModelSpec(
    key="llama-3.1-8b-instruct", family="llama", model_type="llama",
    hf_repo="meta-llama/Llama-3.1-8B-Instruct", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.43.0", tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.layers"),
    caveats=(
        "cleanest arch; also loads in TransformerLens",
        "tool-call format differs from Qwen/Hermes — add a Llama codec before agent use",
    ),
))
_add(ModelSpec(
    key="gemma-3-1b-it", family="gemma3", model_type="gemma3",
    hf_repo="google/gemma-3-1b-it", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.50.0",
    module_paths=ModulePaths(decoder_layers="model.layers"),
    caveats=("text-only size; tied embeddings by default",),
))

# ----------------------------------------------------------------------
# VLMs (vision tower + TokenTypeMap)
# ----------------------------------------------------------------------
_add(ModelSpec(
    key="qwen3-vl-8b-instruct", family="qwen3_vl", model_type="qwen3_vl",
    hf_repo="Qwen/Qwen3-VL-8B-Instruct", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.57.0", tool_calling=True,
    module_paths=ModulePaths(
        decoder_layers="model.language_model.layers", vision_tower="model.visual",
        vision_blocks="model.visual.blocks"),
    vision=VisionSpec(image_token_id_attr="image_token_id",
                      merge_size_attr="vision_config.spatial_merge_size", grid_source="grid_thw"),
    caveats=(
        "single .model (model.language_model.layers / model.visual) — verified vs transformers main",
        "DeepStack injects vision residuals into early text layers; 'embedding at pos p' not a single tensor",
        "pop token_type_ids before generate; interleaved-MRoPE 3D position_ids",
    ),
))
_add(ModelSpec(
    key="glm-4.5v", family="glm4v_moe", model_type="glm4v_moe",
    hf_repo="zai-org/GLM-4.5V", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.57.1",
    is_moe=True, is_reasoning=True, tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.language_model.layers", vision_tower="model.visual"),
    vision=VisionSpec(image_token_id_attr="image_token_id"),
    caveats=("106B MoE needs multi-GPU/FP8", "<think> on by default", "read image_token_id from config (151363)"),
))
_add(ModelSpec(
    key="glm-4.1v-9b-thinking", family="glm4v", model_type="glm4v",
    hf_repo="THUDM/GLM-4.1V-9B-Thinking", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.57.0", is_reasoning=True, tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.language_model.layers", vision_tower="model.visual"),
    vision=VisionSpec(image_token_id_attr="image_token_id"),
    caveats=("dense 40L; image_token_id=151343 (≠ GLM-4.5V) — always read from config",),
))
_add(ModelSpec(
    key="kimi-vl-a3b-thinking", family="kimi_vl", model_type="kimi_vl",
    hf_repo="moonshotai/Kimi-VL-A3B-Thinking-2506", auto_class="AutoModelForCausalLM",
    processor_class="AutoProcessor", trust_remote_code=True, min_transformers="4.51.0",
    is_moe=True, is_reasoning=True, attn_semantics=AttnSemantics.MLA_LATENT,
    module_paths=ModulePaths(
        decoder_layers="language_model.model.layers", vision_tower="vision_tower",
        vision_blocks="vision_tower.encoder.blocks"),
    vision=VisionSpec(image_token_id_attr="media_placeholder_token_id", grid_source="grid_hw"),
    caveats=(
        "permanent remote-code; LM is an embedded DeepseekV3ForCausalLM (MLA)",
        "MoonViT vision has NO self_attn (fused wqkv + functional attn) — can't use output_attentions there",
        "no outer .model: language_model.model.layers",
    ),
))
_add(ModelSpec(
    key="llama-4-scout", family="llama4", model_type="llama4",
    hf_repo="meta-llama/Llama-4-Scout-17B-16E-Instruct", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.51.0", is_moe=True,
    module_paths=ModulePaths(
        decoder_layers="language_model.model.layers", vision_tower="vision_model"),
    vision=VisionSpec(image_token_id_attr="image_token_index"),
    caveats=(
        "no outer .model (language_model.model.layers)", "fused experts; iRoPE NoPE layers; chunked attention",
        "Llama tool-call format ≠ Qwen/Hermes — add a Llama codec before agent use",
    ),
))
_add(ModelSpec(
    key="step-1o-vision", family="step", model_type="step1o",
    hf_repo="", auto_class="", api_only=True, attn_semantics=AttnSemantics.NONE,
    caveats=("closed weights; api backend only; no InternalsHandle. Open white-box path: step3-vl-10b.",),
))

__all__ = ["REGISTRY", "get_spec", "list_specs"]
