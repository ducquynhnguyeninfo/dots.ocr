import os
import sys
import json
import argparse
import time
import platform
from datetime import datetime, timezone
from pathlib import Path
import gc

from dots_ocr import DotsOCRParser
from dots_ocr.utils.consts import image_extensions

try:
    import psutil  # type: ignore[import]
except ImportError:  # pragma: no cover
    psutil = None

try:
    import GPUtil  # type: ignore[import]
except ImportError:  # pragma: no cover
    GPUtil = None

# Force CUDA to use PCI bus ordering (same as nvidia-smi)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# Now select GPU 0 (which will match nvidia-smi GPU 0 = RTX 5060 Ti 16GB)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def iter_inputs(input_dir: Path):
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in image_extensions or ext == ".pdf":
            yield p


def get_system_metrics():
    """Capture CPU, RAM, and disk usage stats using psutil when available."""
    metrics = {}
    if not psutil:
        return metrics

    # Average CPU usage since the last call.
    metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
    virtual_memory = psutil.virtual_memory()
    metrics.update({
        "ram_total": virtual_memory.total,
        "ram_available": virtual_memory.available,
        "ram_percent": virtual_memory.percent,
    })

    disk_usage = psutil.disk_usage(os.getcwd())
    metrics.update({
        "disk_total": disk_usage.total,
        "disk_used": disk_usage.used,
        "disk_free": disk_usage.free,
        "disk_percent": disk_usage.percent,
    })
    return metrics


def get_gpu_metrics():
    """Return a snapshot of VRAM and GPU load using GPUtil if installed."""
    if not GPUtil:
        return {"gpu_available": False}

    gpus = GPUtil.getGPUs()
    if not gpus:
        return {"gpu_available": False}

    primary_gpu = gpus[0]
    total_bytes = int(primary_gpu.memoryTotal * 1024 * 1024)
    used_bytes = int(primary_gpu.memoryUsed * 1024 * 1024)

    return {
        "gpu_available": True,
        "gpu_index": primary_gpu.id,
        "gpu_name": primary_gpu.name,
        "gpu_count": len(gpus),
        "gpu_total_memory": total_bytes,
        "gpu_used_memory": used_bytes,
        "gpu_memory_percent": primary_gpu.memoryUtil * 100,
        "gpu_load_percent": primary_gpu.load * 100,
        "gpu_temperature": primary_gpu.temperature,
    }


def collect_hardware_snapshot():
    """Compose a complete hardware snapshot with timestamp and resource data."""
    snapshot = {
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
    }
    snapshot.update(get_system_metrics())
    snapshot.update(get_gpu_metrics())
    return snapshot


def log_hardware_metrics(log_file, entry):
    """Persist a single hardware metric entry to the dedicated log file."""
    if log_file is None:
        return
    log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log_file.flush()


def extract_from_path(
    parser: DotsOCRParser,
    input_path: Path,
    prompt_mode: str,
    output_dir: Path,
    save_pages_dir: Path,
    hardware_log=None,
):
    # Track when parsing started so we can calculate processing duration.
    start_time = time.perf_counter()
    results = parser.parse_file(str(input_path), output_dir=str(output_dir), prompt_mode=prompt_mode)
    duration = time.perf_counter() - start_time
    hardware_snapshot = collect_hardware_snapshot()
    hardware_snapshot_data = {
        key: value for key, value in hardware_snapshot.items() if key != "snapshot_time"
    }
    rows = []
    for r in results:
        layout_json_path = r.get("layout_info_path")
        if not layout_json_path:
            # Non-layout prompts may not produce layout json
            continue
        try:
            with open(layout_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # If the saved file isn't valid JSON (e.g., filtered/raw text), skip
            continue

        if not isinstance(data, list):
            # Expecting a list of cells
            continue

        # Save per-page cells JSON
        base = input_path.stem
        page_no = int(r.get("page_no", 0))
        per_doc_dir = save_pages_dir / base
        per_doc_dir.mkdir(parents=True, exist_ok=True)
        # File naming: image -> <image_name>.json; pdf -> <pdf_name>_page_<N>.json
        if input_path.suffix.lower() in image_extensions:
            filename = f"{base}.json"
        else:
            filename = f"{base}_page_{page_no + 1}.json"
        per_page_path = per_doc_dir / filename
        try:
            with open(per_page_path, "w", encoding="utf-8") as w:
                json.dump(data, w, ensure_ascii=False)
        except Exception:
            pass

        if hardware_log is not None:
            log_entry = {
                "processed_at": hardware_snapshot.get("snapshot_time"),
                "file_path": r.get("file_path", str(input_path)),
                "layout_json_path": layout_json_path,
                "page_no": page_no,
                "prompt_mode": prompt_mode,
                "processing_duration_seconds": duration,
                "cells_extracted": len(data),
                "hardware_snapshot": hardware_snapshot_data,
            }
            log_hardware_metrics(hardware_log, log_entry)

        for cell in data:
            bbox = cell.get("bbox")
            category = cell.get("category")
            text = cell.get("text")
            if bbox is None or category is None:
                continue
            rows.append({
                "file_path": r.get("file_path", str(input_path)),
                "page_no": r.get("page_no", 0),
                "text_type": category,
                "bbox": bbox,
                "text": text,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract text/type/bbox from images in a folder using DotsOCRParser")
    parser.add_argument("--input_dir", type=str, default="data/10/1", help="Directory containing images")
    parser.add_argument("--output_jsonl", type=str, default="./output/extracted_text_layout.jsonl", help="Path to output JSONL")
    parser.add_argument("--parser_output", type=str, default="./output", help="Where per-image parser outputs are saved")
    parser.add_argument("--save_pages_dir", type=str, default="./output/pages", help="Directory to write per-page JSON files")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_completion_tokens", type=int, default=16384)
    parser.add_argument("--num_thread", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--use_hf", action="store_true", help="Use local HF weights instead of vLLM server")
    parser.add_argument("--prompt_mode", type=str, default="prompt_layout_all_en", help="Prompt to use (layout prompt recommended)")
    parser.add_argument("--hardware_log", type=str, default="./output/hardware_metrics.jsonl", help="Path for per-file hardware log JSONL")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.parser_output)
    save_pages_dir = Path(args.save_pages_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pages_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    hardware_log_path = Path(args.hardware_log)
    hardware_log_path.parent.mkdir(parents=True, exist_ok=True)

    ocr = DotsOCRParser(
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir=str(output_dir),
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_hf=bool(args.use_hf),
    )

    total_rows = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out, open(hardware_log_path, "w", encoding="utf-8") as hardware_log:
        for path in iter_inputs(input_dir):
            rows = extract_from_path(ocr, path, args.prompt_mode, output_dir, save_pages_dir, hardware_log=hardware_log)
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1

    print(f"Done. Wrote {total_rows} rows to {args.output_jsonl}")


def main_working_dirs():
    working_dirs = [
        # {'name': '10/1', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/10/1', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/10/1'},
        # {'name': '10/2', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/10/2', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/10/2'},
        # {'name': '11/1', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/11/1', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/11/1'},
        # {'name': '11/2', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/11/2', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/11/2'},
        # {'name': '12/1', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/12/1', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/12/1'},
        # {'name': '12/2', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/12/2', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/12/2'}
        {'name': 'all-classes', 'input_dir': '/home/ndquynh/workspace/raw/combined', 'output_dir': '/home/ndquynh/workspace/raw/combined/output/dotsocr_v1.5/'}
    ]

    for working_dir in working_dirs:
        input_dir = Path(working_dir['input_dir'])
        output_jsonl = Path(f'{working_dir["output_dir"]}/extracted_text_layout.jsonl')
        output_dir = Path(f'{working_dir["output_dir"]}')
        save_pages_dir = Path(f'{working_dir["output_dir"]}/output/pages')

        output_dir.mkdir(parents=True, exist_ok=True)
        save_pages_dir.mkdir(parents=True, exist_ok=True)
        Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

        ocr = DotsOCRParser(
            ip='localhost',
            port=8000,
            model_name='model',
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=16384,
            num_thread=8,
            dpi=200,
            output_dir=str(output_dir),
            min_pixels=None,
            max_pixels=None,
            use_hf=True,
        )

        hardware_log_path = output_dir / "hardware_metrics.jsonl"
        hardware_log_path.parent.mkdir(parents=True, exist_ok=True)
        total_rows = 0
        with open(output_jsonl, "w", encoding="utf-8") as out, open(hardware_log_path, "w", encoding="utf-8") as hardware_log:
            print(f"Processing {working_dir['name']}")
            for path in iter_inputs(input_dir):
                print(f"Processing {path}")
                rows = extract_from_path(ocr, path, "prompt_layout_all_en", output_dir, save_pages_dir, hardware_log=hardware_log)
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_rows += 1
        print(f"Done. Wrote {total_rows} rows to {output_jsonl}")
        del ocr
        gc.collect()


if __name__ == "__main__":
    # main()
    main_working_dirs()


