"""Isolated BETO encoder smoke test. No sentiment classifier or DEMO import.

Downloads are explicit, the checkpoint is immutable, remote code is disabled,
and the upstream legacy tensor file is loaded with torch weights_only=True.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import threading
import time
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

MODEL = "dccuchile/bert-base-spanish-wwm-cased"
REVISION = "c4d86612f51b4f46759c8390d1798c2febe71b93"
WEIGHT_SHA256 = "e131a95091c777bbd45250fb647fec415010cb8cd1ad6e1d59babeb82a0be360"
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt", "pytorch_model.bin", "README.md")
PROBE_TEXTS = ("La visita empezó a las nueve y terminó al mediodía.",
               "El recorrido no fue tan bueno como esperaba.")


class EncoderProbeError(ValueError):
    pass


def choose_memory_measurement(parameter_mib, sampled_mib, resource_mib, errors):
    """Reject inconsistent process telemetry rather than recommend impossible RAM."""
    valid_resource = resource_mib is not None and resource_mib >= parameter_mib
    valid_sample = not errors and sampled_mib >= parameter_mib
    return {
        "peak_rss_sampled_mib": sampled_mib if valid_sample else None,
        "peak_rss_verified_mib": resource_mib if valid_resource else sampled_mib if valid_sample else None,
        "peak_rss_method": "Linux getrusage self high-water" if valid_resource else "psutil sampled" if valid_sample else "PENDING_UNRELIABLE_TELEMETRY",
    }


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _private_target(path: Path) -> Path:
    root, target = Path.cwd().resolve(), Path(path).resolve()
    if not any(target.is_relative_to(root / name) for name in ("artifacts", "data/private")):
        raise EncoderProbeError("La caché y el informe requieren artifacts o data/private.")
    return target


def checkpoint(cache: Path, *, download: bool = False) -> dict:
    cache = _private_target(cache)
    manifest_path = cache / "verified_checkpoint.json"
    if not manifest_path.exists():
        if not download:
            raise EncoderProbeError("CACHE_MISS: usar --download explícitamente para obtener el checkpoint.")
        url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
        with urlopen(url, timeout=30) as response:
            info = json.load(response)
        if info.get("sha") != REVISION or info.get("gated"):
            raise EncoderProbeError("Revisión o acceso del checkpoint inesperados.")
        items = {x["rfilename"]: x for x in info["siblings"]}
        cache.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in FILES:
            item, target = items[name], cache / name
            if item.get("size", 0) > 600_000_000:
                raise EncoderProbeError("Archivo de checkpoint mayor que el límite previsto.")
            if not target.exists():
                temporary = cache / (name + ".part")
                created = False
                try:
                    with temporary.open("xb") as output:
                        created = True
                        with urlopen(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", timeout=45) as response:
                            count = 0
                            while block := response.read(1 << 20):
                                count += len(block)
                                if count > 600_000_000:
                                    raise EncoderProbeError("Descarga excede el límite previsto.")
                                output.write(block)
                    temporary.rename(target)
                finally:
                    if created:
                        temporary.unlink(missing_ok=True)
            digest = file_sha(target)
            if target.stat().st_size != item["size"]:
                raise EncoderProbeError("Tamaño del checkpoint no coincide con el origen.")
            if "lfs" in item:
                if digest != item["lfs"]["sha256"]:
                    raise EncoderProbeError("Hash LFS del checkpoint no coincide.")
            else:
                data = target.read_bytes()
                git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                if git_blob != item["blobId"]:
                    raise EncoderProbeError("Hash Git de archivo de configuración no coincide.")
            hashes[name] = {"sha256": digest, "bytes": target.stat().st_size}
        manifest = {"model": MODEL, "revision": REVISION, "files": hashes}
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != MODEL or manifest.get("revision") != REVISION:
        raise EncoderProbeError("Caché de otra revisión o modelo.")
    if set(manifest.get("files", {})) != set(FILES):
        raise EncoderProbeError("Caché incompleta.")
    for name, record in manifest["files"].items():
        if file_sha(cache / name) != record["sha256"]:
            raise EncoderProbeError("Caché modificada.")
    if manifest["files"]["pytorch_model.bin"]["sha256"] != WEIGHT_SHA256:
        raise EncoderProbeError("Pesos diferentes de la revisión fijada.")
    return manifest


def run_probe(cache: Path, output: Path, *, download: bool = False, threads: int = 2) -> dict:
    output = _private_target(output)
    if output.suffix != ".json" or output.exists():
        raise EncoderProbeError("Usar un informe JSON nuevo; no se sobrescriben resultados.")
    if not 1 <= threads <= 8:
        raise EncoderProbeError("Número de hilos fuera de 1–8.")
    manifest = checkpoint(cache, download=download)
    import psutil
    process = None
    peak = [0]
    memory_errors = []
    stop = threading.Event()

    def observe():
        while not stop.wait(.02):
            try:
                if process is not None:
                    peak[0] = max(peak[0], process.memory_info().rss)
            except (psutil.Error, OSError):
                memory_errors.append("PROCESS_MEMORY_UNAVAILABLE")
                return

    monitor = threading.Thread(target=observe, daemon=True)
    monitor.start()
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        try:
            process = psutil.Process(os.getpid())
            peak[0] = process.memory_info().rss
        except (psutil.Error, OSError):
            memory_errors.append("PID_NAMESPACE_NOT_OBSERVABLE")
        if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
            raise EncoderProbeError("El archivo tensor legacy requiere torch>=2.6 y weights_only=True.")
        torch.set_num_threads(threads)
        tokenizer = AutoTokenizer.from_pretrained(str(cache), local_files_only=True, trust_remote_code=False)
        start = time.perf_counter()
        model = AutoModel.from_pretrained(
            str(cache), local_files_only=True, trust_remote_code=False,
            use_safetensors=False, weights_only=True, add_pooling_layer=False,
        ).eval()
        load_seconds = time.perf_counter() - start
        parameter_count = sum(p.numel() for p in model.parameters())
        inputs = tokenizer(list(PROBE_TEXTS), padding=True, truncation=True, max_length=128,
                           return_tensors="pt")
        timings = []
        with torch.inference_mode():
            encoded = model(**inputs).last_hidden_state
            for _ in range(5):
                start = time.perf_counter_ns()
                batch = tokenizer(list(PROBE_TEXTS), padding=True, truncation=True,
                                  max_length=128, return_tensors="pt")
                encoded = model(**batch).last_hidden_state
                timings.append((time.perf_counter_ns() - start) / 1e6)
        if encoded.shape[:2] != inputs["input_ids"].shape or not torch.isfinite(encoded).all():
            raise EncoderProbeError("Salida del encoder inválida.")
        hidden = int(encoded.shape[-1])
        long_tokens = tokenizer("Observación del recorrido. " * 200, truncation=True,
                                max_length=128)["input_ids"]
        resource_peak = None
        if platform.system() == "Linux":
            import resource
            resource_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        sampled_peak = peak[0] / 2**20
        parameter_mib = parameter_count * 4 / 2**20
        # An impossible resident estimate must not become a RAM recommendation.
        memory = choose_memory_measurement(parameter_mib, sampled_peak, resource_peak, memory_errors)
        report = {
            "report_type": "BETO_ENCODER_SMOKE_NOT_SENTIMENT_CLASSIFIER",
            "checkpoint": manifest, "python": platform.python_version(),
            "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "psutil")},
            "device_used": "cpu", "cuda_available": torch.cuda.is_available(), "threads": threads,
            "parameter_count": parameter_count, "parameter_fp32_mib": parameter_count * 4 / 2**20,
            "full_finetune_adam_parameter_gradient_moments_lower_bound_mib": parameter_count * 16 / 2**20,
            "load_seconds": load_seconds, **memory,
            "memory_diagnostic": {"raw_sampled_mib": sampled_peak, "raw_resource_mib": resource_peak, "errors": memory_errors},
            "rss_sampling_interval_ms": 20, "hardware_memory_total_mib": psutil.virtual_memory().total / 2**20,
            "encoder_output_shape": list(encoded.shape), "all_finite": True,
            "token_counts_including_special": inputs["attention_mask"].sum(dim=1).tolist(),
            "unknown_token_counts": [(row == tokenizer.unk_token_id).sum().item() for row in inputs["input_ids"]],
            "long_input_truncated_token_count": len(long_tokens),
            "batch_two_encode_tokenize_ms": timings, "timing_scope": "one warmup; five CPU batch-two forward passes including tokenizer",
            "future_three_class_head": {"parameters": hidden * 3 + 3, "status": "UNTRAINED_NOT_CREATED_OR_EVALUATED", "requires": "supervised training and independent evaluation"},
            "license_stated_intent": "CC BY 4.0 per upstream model-card disclaimer, not a complete original-corpus rights clearance",
            "license_caveat": "Authors note uncertainty about compatibility of all pretraining text licenses, especially commercial use.",
            "limits": ["No GPU or fine-tuning memory/latency measured.", "Optimizer memory is a parameter-only lower bound, not total training RAM.", "No sentiment logits, sentiment probabilities or sentiment accuracy produced.", "No remote code; checkpoint loaded locally with weights_only=True.", "Probe inputs are illustrative, not gold evaluation data."],
        }
    finally:
        stop.set()
        monitor.join(timeout=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/beto-encoder-smoke-v1/checkpoint"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/beto-encoder-smoke-v1/report.json"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = run_probe(args.cache, args.output, download=args.download, threads=args.threads)
    except (EncoderProbeError, ImportError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Encoder probe stopped: {type(exc).__name__}; inspect dependencies/cache locally.\n")
    print(json.dumps({k: report[k] for k in ("report_type", "device_used", "encoder_output_shape", "peak_rss_verified_mib", "peak_rss_method")}, indent=2))


if __name__ == "__main__":
    main()
