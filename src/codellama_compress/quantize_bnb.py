from __future__ import annotations

from pathlib import Path

from .config import save_json


def write_bnb_bundle(*, model_id_or_dir: str, out_dir: Path) -> None:
    """
    bitsandbytes quantization is a load-time optimization, not a stable "export" format.

    This writes a small bundle describing how to load the model using Transformers + bitsandbytes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        out_dir / "bnb_load.json",
        {
            "model": model_id_or_dir,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
        },
    )
    (out_dir / "README.txt").write_text(
        "Load with:\n"
        "  from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "  import torch\n"
        "  tok = AutoTokenizer.from_pretrained(MODEL)\n"
        "  model = AutoModelForCausalLM.from_pretrained(\n"
        "      MODEL,\n"
        "      device_map='auto',\n"
        "      load_in_4bit=True,\n"
        "      torch_dtype=torch.float16,\n"
        "  )\n"
    )
