#!/usr/bin/env python3
"""Memory-conscious, author-protocol V3Det bbox AP and frequency-group AP.

This intentionally reproduces experiments/v3det/eval_3.py: ordinary
pycocotools COCOeval, without the official V3Det hierarchy/parent ignore rule.
AP uses area=all and the largest maxDets entry (300 by default).

Predictions must be one COCO results JSON array, with absolute category IDs.
They are streamed into temporary category-block JSONL files, then evaluated
one category at a time with unmodified COCOeval. Empty image/category pairs
are omitted; their evaluateImg result is None and they cannot affect AP.
Detection order and ascending image order are retained, including score ties.
"""

import argparse
import contextlib
import gc
import io
import json
import math
import os
import tempfile
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


EVALUATOR = "legacy_author_pycocotools_no_hierarchy"


def iter_json_array(path, read_size=1024 * 1024):
    """Read a top-level JSON array incrementally using the standard library."""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8-sig") as stream:
        buffer, position, eof = "", 0, False

        def refill():
            nonlocal buffer, position, eof
            chunk = stream.read(read_size)
            buffer = buffer[position:] + chunk
            position = 0
            eof = not chunk

        def whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                refill()

        whitespace()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError("Predictions must be one top-level COCO JSON array")
        position += 1
        whitespace()
        if position < len(buffer) and buffer[position] == "]":
            position += 1
        else:
            while True:
                whitespace()
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise ValueError("Incomplete or invalid predictions JSON")
                        refill()
                position = end
                yield value
                whitespace()
                if position >= len(buffer):
                    raise ValueError("Unterminated predictions JSON array")
                delimiter = buffer[position]
                position += 1
                if delimiter == "]":
                    break
                if delimiter != ",":
                    raise ValueError("Expected comma or closing bracket in predictions")
        whitespace()
        if position != len(buffer):
            raise ValueError("Unexpected content after predictions JSON array")


def load_groups(path, category_ids):
    with open(path, encoding="utf-8") as stream:
        document = json.load(stream)
    groups = document.get("groups", document)
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Groups must be an object: {\"groups\": {name: [category_ids]}}")
    known, seen, normalized = set(category_ids), set(), {}
    for name, values in groups.items():
        if not isinstance(values, list):
            raise ValueError("Each groups value must be a list of category IDs")
        ids = [int(value) for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated category ID in group " + str(name))
        unknown, overlap = set(ids) - known, set(ids) & seen
        if unknown or overlap:
            raise ValueError("Unknown or overlapping group category IDs: " + str(unknown or overlap))
        seen.update(ids)
        normalized[str(name)] = sorted(ids)
    if known - seen:
        raise ValueError("Groups must partition all annotation categories; missing " + str(len(known - seen)))
    return normalized


def coco_from_parts(images, categories, annotations):
    coco = COCO()
    coco.dataset = {"info": {}, "images": images, "categories": categories,
                    "annotations": annotations}
    coco.createIndex()
    return coco


def evaluate_category(category, annotations, predictions, image_lookup, max_dets):
    """Return standard precision for area=all at maxDets[-1]."""
    category_id = int(category["id"])
    image_ids = sorted({int(a["image_id"]) for a in annotations}
                       | {int(p["image_id"]) for p in predictions})
    images = [image_lookup[i] for i in image_ids]
    if not image_ids:
        return np.full((10, 101), -1.0)
    coco_gt = coco_from_parts(images, [category], annotations)
    if predictions:
        # loadRes fills standard bbox area/segmentation/iscrowd fields and
        # preserves input order, just as the original author's evaluator does.
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = coco_from_parts(images, [category], [])
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.catIds = [category_id]
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = list(max_dets)
    # Other area ranges do not participate in APb/AP50/AP75. Keeping only
    # 'all' saves memory without changing any reported metric.
    evaluator.params.areaRng = [[0, 1e10]]
    evaluator.params.areaRngLbl = ["all"]
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator.eval["precision"][:, :, 0, 0, -1].copy()


def precision_stats(precision):
    output = {}
    for name, values in (("AP", precision), ("AP50", precision[0]),
                         ("AP75", precision[5])):
        valid = values[values > -1]
        output[name] = {"sum": float(valid.sum()), "count": int(valid.size)}
    return output


def aggregate(records, ids):
    totals = {name: [0.0, 0] for name in ("AP", "AP50", "AP75")}
    for category_id in ids:
        for name in totals:
            stats = records[category_id]["precision_stats"][name]
            totals[name][0] += stats["sum"]
            totals[name][1] += stats["count"]
    return {
        "total_categories": len(ids),
        "valid_categories": sum(records[i]["precision_stats"]["AP"]["count"] > 0 for i in ids),
        "gt_annotations": sum(records[i]["gt_annotations"] for i in ids),
        "predictions": sum(records[i]["predictions"] for i in ids),
        **{name: total / count * 100 if count else None
           for name, (total, count) in totals.items()},
    }


def spool_predictions(path, category_ids, image_ids, directory, chunk_size):
    category_to_chunk = {category_id: index // chunk_size
                         for index, category_id in enumerate(category_ids)}
    handles = OrderedDict()
    counts = defaultdict(int)
    images_with_predictions = set()
    try:
        for number, prediction in enumerate(iter_json_array(path), 1):
            if not isinstance(prediction, dict):
                raise ValueError("Each prediction must be an object")
            category_id, image_id = prediction.get("category_id"), prediction.get("image_id")
            if type(category_id) is not int or category_id not in category_to_chunk:
                raise ValueError("Unknown/non-integer prediction category_id: " + str(category_id))
            if type(image_id) is not int or image_id not in image_ids:
                raise ValueError("Unknown/non-integer prediction image_id: " + str(image_id))
            bbox, score = prediction.get("bbox"), prediction.get("score")
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
                raise ValueError("Prediction bbox must be four finite numbers")
            if bbox[2] < 0 or bbox[3] < 0:
                raise ValueError("Prediction bbox width/height must not be negative")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("Prediction score must be finite")
            chunk = category_to_chunk[category_id]
            if chunk not in handles:
                if len(handles) >= 32:
                    _, handle = handles.popitem(last=False)
                    handle.close()
                handles[chunk] = open(Path(directory) / (str(chunk) + ".jsonl"),
                                      "a", encoding="utf-8", buffering=65536)
            handles.move_to_end(chunk)
            # Only COCO bbox result fields are needed. Preserve their exact
            # Python float values and prediction order through JSON roundtrip.
            compact = {"image_id": image_id, "category_id": category_id,
                       "bbox": bbox, "score": score}
            handles[chunk].write(json.dumps(compact, separators=(",", ":")) + "\n")
            counts[category_id] += 1
            images_with_predictions.add(image_id)
            if number % 1000000 == 0:
                print("Streamed {:,} predictions".format(number), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    return counts, len(images_with_predictions)


def run(args):
    started = time.time()
    with open(args.annotations, encoding="utf-8") as stream:
        ground_truth = json.load(stream)
    categories = {int(c["id"]): c for c in ground_truth["categories"]}
    images = {int(i["id"]): i for i in ground_truth["images"]}
    if len(categories) != len(ground_truth["categories"]) or len(images) != len(ground_truth["images"]):
        raise ValueError("Duplicate category/image IDs in annotations")
    category_ids = sorted(categories)
    groups = load_groups(args.groups, category_ids)
    annotations_by_category = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        if annotation["category_id"] not in categories or annotation["image_id"] not in images:
            raise ValueError("GT annotation references unknown image/category")
        annotations_by_category[int(annotation["category_id"])].append(annotation)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch_parent = Path(args.scratch_dir) if args.scratch_dir else output.parent
    scratch_parent.mkdir(parents=True, exist_ok=True)
    records = {}
    with tempfile.TemporaryDirectory(prefix="coco_group_eval_", dir=str(scratch_parent)) as scratch:
        counts, images_with_predictions = spool_predictions(
            args.predictions, category_ids, set(images), scratch, args.category_chunk_size)
        print("Evaluating {:,} categories, {:,} GT images, {:,} predictions".format(
            len(category_ids), len(images), sum(counts.values())), flush=True)
        for offset in range(0, len(category_ids), args.category_chunk_size):
            chunk_ids = category_ids[offset:offset + args.category_chunk_size]
            predictions_by_category = defaultdict(list)
            shard = Path(scratch) / (str(offset // args.category_chunk_size) + ".jsonl")
            if shard.exists():
                with open(shard, encoding="utf-8") as stream:
                    for line in stream:
                        prediction = json.loads(line)
                        predictions_by_category[prediction["category_id"]].append(prediction)
            for category_id in chunk_ids:
                annotations = annotations_by_category[category_id]
                predictions = predictions_by_category.pop(category_id, [])
                with contextlib.redirect_stdout(io.StringIO()):
                    precision = evaluate_category(categories[category_id], annotations,
                                                  predictions, images, args.max_dets)
                stats = precision_stats(precision)
                records[category_id] = {
                    "category_id": category_id,
                    "name": categories[category_id].get("name"),
                    "gt_annotations": len(annotations),
                    "predictions": counts[category_id],
                    "precision_stats": stats,
                    **{name: value["sum"] / value["count"] * 100 if value["count"] else None
                       for name, value in stats.items()},
                }
            if shard.exists():
                shard.unlink()
            gc.collect()
            print("Evaluated {}/{} categories; AP={}".format(
                min(offset + args.category_chunk_size, len(category_ids)), len(category_ids),
                aggregate(records, list(records))["AP"]), flush=True)
    result = {
        "evaluator": EVALUATOR,
        "protocol": {
            "iou_type": "bbox", "iou_thresholds": [round(.5 + .05 * i, 2) for i in range(10)],
            "recall_thresholds": "0:0.01:1", "area_range": [0, 1e10],
            "max_dets": args.max_dets, "reported_max_dets": args.max_dets[-1],
            "units": "percent", "hierarchy_ignore": False,
            "missing_category_handling": "Exclude precision=-1; no valid categories gives null",
            "implementation": "Unmodified COCOeval per category; omit empty image/category pairs",
            "group_definition": "User-supplied partition; no claim of official V3Det frequency groups",
        },
        "inputs": {"annotations": str(Path(args.annotations).resolve()),
                   "predictions": str(Path(args.predictions).resolve()),
                   "groups": str(Path(args.groups).resolve())},
        "image_count": len(images), "images_with_predictions": images_with_predictions,
        "prediction_count": sum(counts.values()),
        "overall": aggregate(records, category_ids),
        "groups": {name: aggregate(records, ids) for name, ids in groups.items()},
        "per_category": {str(i): {k: v for k, v in records[i].items() if k != "precision_stats"}
                         for i in category_ids},
        "elapsed_seconds": time.time() - started,
    }
    temporary_output = output.with_name(output.name + ".tmp")
    with open(temporary_output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary_output, output)
    print(json.dumps({"overall": result["overall"], "groups": result["groups"]},
                     ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-dets", nargs="+", type=int, default=[1, 10, 300])
    parser.add_argument("--category-chunk-size", type=int, default=32)
    parser.add_argument("--scratch-dir", help="Temporary JSONL shards; default: output parent")
    args = parser.parse_args()
    if args.category_chunk_size < 1 or any(value < 1 for value in args.max_dets):
        parser.error("Chunk size and maxDets must be positive")
    if args.max_dets != sorted(set(args.max_dets)):
        parser.error("maxDets must be distinct and increasing")
    run(args)


if __name__ == "__main__":
    main()
