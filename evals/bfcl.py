"""BFCL CLI with an extra model "harwyn-FC". BFCL has no Qwen3.5 handler, so this goes through vLLM's
OpenAI tools API, which applies Qwen3.5's own chat template and qwen3_coder tool-call parser.
usage: OPENAI_BASE_URL=http://localhost:8000/v1 python bfcl.py generate --model harwyn-FC --test-category ...
"""
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

MODEL_CONFIG_MAPPING["harwyn-FC"] = ModelConfig(
    model_name="m", display_name="harwyn (FC)", url="", org="", license="",
    model_handler=OpenAICompletionsHandler, input_price=None, output_price=None,
    is_fc_model=True, underscore_to_dot=True)

from bfcl_eval.__main__ import cli  # noqa: E402

cli()
