#!/usr/bin/env python3

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path


def first_value(record, keys):
    for key in keys:
        if key in record:
            value = record[key]
            if value is not None and value != "" and value != []:
                return value
    return None


def as_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return "\n".join(as_text(x) for x in value)

    return json.dumps(value, ensure_ascii=False, indent=2)


def find_records(data):
    if isinstance(data, list):
        return data

    preferred_keys = [
        "predictions",
        "results",
        "samples",
        "records",
        "data",
        "outputs",
    ]

    for key in preferred_keys:
        value = data.get(key)
        if isinstance(value, list):
            return value

    for key, value in data.items():
        if isinstance(value, list):
            print(f"Using list under top-level key: {key}")
            return value

    raise RuntimeError(
        "Could not find a list of individual prediction records."
    )


def resolve_image_path(value, image_root):
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None
        value = value[0]

    value = str(value)

    path = Path(value)

    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(path)

        if image_root:
            candidates.append(Path(image_root) / path)
            candidates.append(Path(image_root) / path.name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None


def image_to_data_uri(path):
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/png"

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="/data/ruida/Qwen3VL_runs/"
                "deeplesion_test_predictions_checkpoint104200.json",
    )

    parser.add_argument(
        "--output",
        default="/data/ruida/Qwen3VL_runs/"
                "deeplesion_test_predictions_checkpoint104200_viewer.html",
    )

    parser.add_argument(
        "--image_root",
        default="",
        help="Optional root directory for relative image paths.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use 0 to include all records.",
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = find_records(data)

    if args.limit > 0:
        records = records[:args.limit]

    image_keys = [
        "image",
        "images",
        "image_path",
        "image_file",
        "filename",
        "file_name",
        "img",
    ]

    reference_keys = [
        "reference",
        "ground_truth",
        "ground_truth_report",
        "gt",
        "gt_report",
        "target",
        "target_text",
        "report",
        "answer",
        "label",
    ]

    prediction_keys = [
        "prediction",
        "predicted_report",
        "pred",
        "generated_report",
        "generated_text",
        "output",
        "response",
        "hypothesis",
    ]

    id_keys = [
        "id",
        "case_id",
        "image_id",
        "sample_id",
        "study_id",
        "uid",
    ]

    sections = []
    images_found = 0
    missing_images = 0

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue

        case_id = first_value(record, id_keys)
        image_value = first_value(record, image_keys)
        reference = first_value(record, reference_keys)
        prediction = first_value(record, prediction_keys)

        case_id = as_text(case_id) or f"case_{index:05d}"
        reference = as_text(reference) or "[Reference not found]"
        prediction = as_text(prediction) or "[Prediction not found]"

        image_path = resolve_image_path(image_value, args.image_root)

        if image_path is not None:
            images_found += 1
            image_html = (
                f'<img src="{image_to_data_uri(image_path)}" '
                f'alt="{html.escape(case_id)}">'
                f'<div class="path">{html.escape(str(image_path))}</div>'
            )
        else:
            missing_images += 1
            image_html = (
                '<div class="missing-image">Image not found</div>'
                f'<div class="path">'
                f'{html.escape(as_text(image_value) or "[No image field]")}'
                f'</div>'
            )

        sections.append(
            f"""
<section class="case">
    <div class="case-header">
        <strong>Case {index}</strong>
        <span>{html.escape(case_id)}</span>
    </div>

    <div class="grid">
        <div class="image-panel">
            <h3>Image</h3>
            {image_html}
        </div>

        <div class="text-panel reference">
            <h3>Ground-truth reference</h3>
            <div class="report">{html.escape(reference)}</div>
        </div>

        <div class="text-panel prediction">
            <h3>Qwen3-VL prediction</h3>
            <div class="report">{html.escape(prediction)}</div>
        </div>
    </div>
</section>
"""
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Qwen3-VL DeepLesion Predictions</title>

<style>
body {{
    margin: 0;
    padding: 24px;
    background: #f4f6f8;
    font-family: Arial, Helvetica, sans-serif;
    color: #1f2933;
}}

h1 {{
    margin-bottom: 8px;
}}

.summary {{
    background: white;
    border: 1px solid #d9dee5;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 24px;
    line-height: 1.6;
}}

.case {{
    background: white;
    border: 1px solid #d9dee5;
    border-radius: 10px;
    margin-bottom: 24px;
    overflow: hidden;
}}

.case-header {{
    background: #e9eef5;
    padding: 12px 16px;
    display: flex;
    gap: 20px;
    align-items: center;
    overflow-wrap: anywhere;
}}

.grid {{
    display: grid;
    grid-template-columns: minmax(260px, 0.8fr) 1fr 1fr;
}}

.image-panel,
.text-panel {{
    padding: 18px;
    border-right: 1px solid #d9dee5;
}}

.text-panel:last-child {{
    border-right: none;
}}

.image-panel img {{
    display: block;
    max-width: 100%;
    max-height: 500px;
    margin: auto;
    object-fit: contain;
    background: black;
}}

.report {{
    white-space: pre-wrap;
    line-height: 1.6;
    font-size: 16px;
}}

.path {{
    margin-top: 10px;
    font-family: monospace;
    font-size: 11px;
    color: #667085;
    overflow-wrap: anywhere;
}}

.missing-image {{
    min-height: 250px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #eeeeee;
    color: #777777;
}}

@media (max-width: 1000px) {{
    .grid {{
        grid-template-columns: 1fr;
    }}

    .image-panel,
    .text-panel {{
        border-right: none;
        border-bottom: 1px solid #d9dee5;
    }}
}}
</style>
</head>

<body>

<h1>Qwen3-VL DeepLesion test results</h1>

<div class="summary">
    <strong>Input:</strong> {html.escape(args.input)}<br>
    <strong>Cases displayed:</strong> {len(records)}<br>
    <strong>Images found:</strong> {images_found}<br>
    <strong>Images missing:</strong> {missing_images}
</div>

{''.join(sections)}

</body>
</html>
"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")

    print("========================================")
    print(f"Cases written  : {len(records)}")
    print(f"Images found   : {images_found}")
    print(f"Images missing : {missing_images}")
    print(f"HTML saved to  : {output}")
    print("========================================")


if __name__ == "__main__":
    main()
