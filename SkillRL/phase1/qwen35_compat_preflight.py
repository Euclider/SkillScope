#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXPECTED = {
    "torch": "2.10.0",
    "transformers": "5.10.4",
    "vllm": "0.19.1",
    "fla-core": "0.5.2",
    "causal-conv1d": "1.7.0",
}


def result(status: str, message: str) -> dict[str, str]:
    return {"status": status, "message": message}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the Qwen3.5 SkillRL compatibility stack")
    parser.add_argument("--model", type=Path, default=Path("/home/wangyifan/model/Qwen3.5-4B"))
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--run-kernel-smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/preflight/qwen35-environment.json"),
    )
    args = parser.parse_args()

    checks: dict[str, dict[str, str]] = {}
    for package, expected in EXPECTED.items():
        try:
            installed = version(package)
            status = "pass" if installed == expected or installed.startswith(expected + "+") else "fail"
            checks[f"package:{package}"] = result(status, f"installed={installed}; expected={expected}")
        except PackageNotFoundError:
            checks[f"package:{package}"] = result("fail", "not installed")

    try:
        import torch
        import causal_conv1d  # noqa: F401
        import fla  # noqa: F401
        from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5ForCausalLM

        mapped = type(Qwen3_5Config()) in AutoModelForCausalLM._model_mapping.keys()
        checks["qwen35_registry"] = result(
            "pass" if mapped else "fail",
            f"AutoModelForCausalLM mapping={mapped}; class={Qwen3_5ForCausalLM.__name__}",
        )
        checks["cuda"] = result(
            "pass" if torch.cuda.is_available() else "fail",
            f"available={torch.cuda.is_available()}; count={torch.cuda.device_count()}; runtime={torch.version.cuda}",
        )

        if args.run_kernel_smoke and torch.cuda.is_available():
            from transformers import Qwen3_5TextConfig

            config = Qwen3_5TextConfig(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                layer_types=["linear_attention", "full_attention"],
                max_position_embeddings=64,
                use_cache=False,
            )
            model = Qwen3_5ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
            input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            output.loss.backward()
            checks["qwen35_kernel_forward_backward"] = result(
                "pass",
                f"loss={output.loss.detach().float().item():.6f}; dtype=bf16",
            )
    except Exception as error:
        checks["qwen35_runtime"] = result("fail", repr(error))

    model_status = "pass" if args.model.is_dir() else ("fail" if args.require_model else "warning")
    checks["local_model"] = result(model_status, f"path={args.model}; exists={args.model.is_dir()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    failures = [name for name, check in checks.items() if check["status"] == "fail"]
    if failures:
        raise SystemExit(f"Qwen3.5 preflight failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
