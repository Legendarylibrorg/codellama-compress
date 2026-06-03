from __future__ import annotations

import shlex
from pathlib import Path

from .security import (
    assert_safe_modelfile_name,
    assert_safe_shell_token,
    clamp_tcp_port,
    resolve_user_path,
)


def write_export_bundle(
    *, model_dir: Path, out_dir: Path, model_name: str, port: int = 8000
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir_r = resolve_user_path(model_dir)
    model_dir_s = assert_safe_shell_token(str(model_dir_r), field="model_dir")
    model_name = assert_safe_modelfile_name(model_name)
    port = clamp_tcp_port(port)
    model_dir_q = shlex.quote(model_dir_s)

    # vLLM server script (OpenAI-compatible)
    (out_dir / "vllm_server.sh").write_text(
        "# GENERATED FILE. Review before running.\n"
        "#!/usr/bin/env bash\n"
        "set -e\n"
        f"MODEL_DIR={model_dir_q}\n"
        f'PORT="{port}"\n'
        "python -m vllm.entrypoints.openai.api_server \\\n"
        '  --model "$MODEL_DIR" \\\n'
        '  --port "$PORT" \\\n'
        "  --dtype float16\n"
    )

    # Dockerfile (GPU runtime)
    (out_dir / "Dockerfile").write_text(
        "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04\n"
        "WORKDIR /app\n"
        "RUN apt-get update && apt-get install -y python3 python3-venv python3-pip && rm -rf /var/lib/apt/lists/*\n"
        "COPY . /app\n"
        "RUN python3 -m pip install --upgrade pip && pip install . && pip install vllm\n"
        f"EXPOSE {port}\n"
        f'CMD ["bash", "output/export/vllm_server.sh"]\n'
    )

    # Dockerfile with optional quant deps (larger supply-chain footprint)
    (out_dir / "Dockerfile.quant").write_text(
        "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04\n"
        "WORKDIR /app\n"
        "RUN apt-get update && apt-get install -y python3 python3-venv python3-pip && rm -rf /var/lib/apt/lists/*\n"
        "COPY . /app\n"
        'RUN python3 -m pip install --upgrade pip && pip install ".[quant]" && pip install vllm\n'
        f"EXPOSE {port}\n"
        f'CMD ["bash", "output/export/vllm_server.sh"]\n'
    )

    # Ollama Modelfile (points at GGUF output produced by convert script)
    (out_dir / "Modelfile").write_text(
        f'FROM ./model-q4_k_m.gguf\nPARAMETER stop "<|eot_id|>"\nSYSTEM "You are {model_name}."\n'
    )

    # GGUF conversion helper (requires llama.cpp checkout)
    (out_dir / "convert_gguf.sh").write_text(
        "# GENERATED FILE. Review before running.\n"
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "_validate_path() {\n"
        '  case "$1" in\n'
        "    ''|*..*|*$'*|*';'*|*'|'*|*'&'*|*'<'*|*'>'*|*$'\\n'*|*$'\\r'*)\n"
        '      echo "Unsafe path argument: $1" >&2\n'
        "      exit 1\n"
        "      ;;\n"
        "  esac\n"
        "}\n"
        'IN_DIR="${1:-' + model_dir_q + '}"\n'
        'OUT_DIR="${2:-output/gguf}"\n'
        'QUANTS="${3:-q4_k_m}"\n'
        'LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"\n'
        '_validate_path "$IN_DIR"\n'
        '_validate_path "$OUT_DIR"\n'
        '_validate_path "$LLAMA_CPP_DIR"\n'
        'mkdir -p "$OUT_DIR"\n'
        'if [ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then\n'
        '  echo "Expected llama.cpp at $LLAMA_CPP_DIR with convert_hf_to_gguf.py" >&2\n'
        '  echo "Clone it: git clone https://github.com/ggerganov/llama.cpp llama.cpp" >&2\n'
        "  exit 1\n"
        "fi\n"
        'python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$IN_DIR" --outtype f16 --outfile "$OUT_DIR/model-f16.gguf"\n'
        "IFS=',' read -ra QLIST <<< \"$QUANTS\"\n"
        'for q in "${QLIST[@]}"; do\n'
        '  "$LLAMA_CPP_DIR/llama-quantize" "$OUT_DIR/model-f16.gguf" "$OUT_DIR/model-${q}.gguf" "$q"\n'
        "done\n"
    )

    # README for the export bundle
    (out_dir / "README.md").write_text(
        "# Export bundle\n\n"
        "This folder contains helper artifacts for serving and exporting.\n\n"
        "## Docker\n\n"
        "- `Dockerfile`: installs the project + `vllm` only (smaller dependency footprint).\n"
        "- `Dockerfile.quant`: also installs `.[quant]` (larger footprint; use only if needed).\n\n"
        "## Speculative decoding\n\n"
        "Run locally via the CLI:\n\n"
        "```bash\n"
        "codellama-compress util speculative \\\n"
        f"  --target-model {model_dir_s} \\\n"
        "  --draft-model codellama/CodeLlama-7b-hf \\\n"
        "  --num-speculative-tokens 5\n"
        "```\n\n"
        "## vLLM\n\n"
        "Requires `vllm` installed.\n\n"
        "```bash\n"
        "bash ./vllm_server.sh\n"
        "```\n\n"
        "## GGUF\n\n"
        "Requires `llama.cpp` (see `convert_gguf.sh`).\n\n"
        "```bash\n"
        'bash ./convert_gguf.sh <hf_model_dir> <out_dir> "q4_k_m,q5_k_m,q8_0"\n'
        "```\n"
    )

    # Convenience python wrapper (matches older README style)
    (out_dir / "speculative_decoding.py").write_text(
        "from codellama_compress.speculative import speculative_generate\n\n"
        "if __name__ == '__main__':\n"
        "    text, stats = speculative_generate(\n"
        f"        prompt='def fibonacci(n):',\n"
        f"        target_model={str(model_dir_r)!r},\n"
        "        draft_model='codellama/CodeLlama-7b-hf',\n"
        "        num_speculative_tokens=5,\n"
        "        max_new_tokens=256,\n"
        "    )\n"
        "    print(text)\n"
        "    print(stats.to_dict())\n"
    )

    # Make scripts executable (best-effort)
    for p in ["vllm_server.sh", "convert_gguf.sh"]:
        try:
            (out_dir / p).chmod(0o755)
        except Exception:
            pass
