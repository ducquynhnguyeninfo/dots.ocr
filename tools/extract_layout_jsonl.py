import os
import sys
import json
import argparse
from pathlib import Path
import gc

from dots_ocr import DotsOCRParser
from dots_ocr.utils.consts import image_extensions


def iter_inputs(input_dir: Path):
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in image_extensions or ext == ".pdf":
            yield p


def extract_from_path(parser: DotsOCRParser, input_path: Path, prompt_mode: str, output_dir: Path, save_pages_dir: Path):
    results = parser.parse_file(str(input_path), output_dir=str(output_dir), prompt_mode=prompt_mode)
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
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.parser_output)
    save_pages_dir = Path(args.save_pages_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pages_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

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
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for path in iter_inputs(input_dir):
            rows = extract_from_path(ocr, path, args.prompt_mode, output_dir, save_pages_dir)
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1

    print(f"Done. Wrote {total_rows} rows to {args.output_jsonl}")


def main_working_dirs():
    working_dirs = [
        # {'name': '10/1', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/10/1', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/10/1'},
        # {'name': '10/2', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/10/2', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/10/2'},
        {'name': '11/1', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/11/1', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/11/1'},
        # {'name': '11/2', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/11/2', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/11/2'},
        # {'name': '12/1', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/12/1', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/12/1'},
        # {'name': '12/2', 'input_dir': '/home/quynhnguyen/dotsocr/dots.ocr/data/12/2', 'output_dir': '/home/quynhnguyen/dotsocr/dots.ocr/output/12/2'}
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

        total_rows = 0
        with open(output_jsonl, "w", encoding="utf-8") as out:
            print(f"Processing {working_dir['name']}")
            for path in iter_inputs(input_dir):
                print(f"Processing {path}")
                rows = extract_from_path(ocr, path, "prompt_layout_all_en", output_dir, save_pages_dir)
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_rows += 1
        print(f"Done. Wrote {total_rows} rows to {output_jsonl}")
        del ocr
        gc.collect()


if __name__ == "__main__":
    # main()
    main_working_dirs()


