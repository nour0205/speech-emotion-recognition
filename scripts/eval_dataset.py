#!/usr/bin/env python3
"""Evaluate emotion recognition on a folder-based dataset.

This script evaluates the SER model on a dataset organized as:
    dataset_root/
        angry/*.wav
        happy/*.wav
        sad/*.wav
        neutral/*.wav
        ...

It reports:
- Per-class accuracy
- Macro F1 score
- Overall accuracy
- Confusion matrix (optional)

Usage:
    python scripts/eval_dataset.py --data_root /path/to/dataset
    python scripts/eval_dataset.py --data_root /path/to/dataset --limit_per_class 100
    python scripts/eval_dataset.py --data_root /path/to/dataset --model baseline --confusion

Example output:
    Evaluation Results
    ==================
    Total samples: 400
    Overall accuracy: 78.5%
    Macro F1: 0.76
    
    Per-class results:
    ------------------
    angry:   80.0% (80/100)
    happy:   75.0% (75/100)
    sad:     82.0% (82/100)
    neutral: 77.0% (77/100)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm

from model import predict_clip, CANONICAL_LABELS
from model.errors import ModelError
from audioio.errors import AudioIOError


class ClassMetrics(TypedDict):
    """Metrics for a single class."""
    true_positives: int
    false_positives: int
    false_negatives: int
    total: int
    correct: int


def compute_precision_recall_f1(
    tp: int, fp: int, fn: int
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score.
    
    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        
    Returns:
        Tuple of (precision, recall, f1).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def normalize_label(label: str) -> str:
    """Normalize a label to canonical form.
    
    Handles common variations like 'ang' -> 'angry', 'hap' -> 'happy'.
    """
    label = label.lower().strip()
    
    # Common mappings from folder names to canonical labels
    mappings = {
        # IEMOCAP short codes
        "ang": "angry",
        "hap": "happy",
        "neu": "neutral",
        "sad": "sad",
        # Full names
        "angry": "angry",
        "happy": "happy",
        "sad": "sad",
        "neutral": "neutral",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",
        # Common variations
        "anger": "angry",
        "happiness": "happy",
        "sadness": "sad",
        "surprised": "surprise",
        "fearful": "fear",
        "disgusted": "disgust",
    }
    
    return mappings.get(label, label)


def find_audio_files(directory: Path, extensions: set[str] = {".wav", ".mp3", ".flac"}) -> list[Path]:
    """Find all audio files in a directory."""
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(files)


def main() -> int:
    """Main entry point for the evaluation script.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        description="Evaluate emotion recognition on a folder-based dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset structure:
    dataset_root/
        angry/*.wav
        happy/*.wav
        sad/*.wav
        neutral/*.wav

Examples:
    %(prog)s --data_root ./data/emotions
    %(prog)s --data_root ./data/emotions --limit_per_class 50
    %(prog)s --data_root ./data/emotions --confusion --json output.json
        """,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the dataset with subdirectories per emotion",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="baseline",
        help="Model ID to use for inference (default: baseline)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        help="Device to run inference on (cpu or cuda, default: cpu)",
    )
    parser.add_argument(
        "--limit_per_class", "-l",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per class (default: all)",
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="Print confusion matrix",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Validate data root
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}", file=sys.stderr)
        return 1
    
    if not data_root.is_dir():
        print(f"Error: Data root is not a directory: {data_root}", file=sys.stderr)
        return 1
    
    # Find emotion subdirectories
    emotion_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    if not emotion_dirs:
        print(f"Error: No subdirectories found in {data_root}", file=sys.stderr)
        return 1
    
    # Collect samples
    samples: list[tuple[Path, str]] = []  # (file_path, true_label)
    label_counts: dict[str, int] = defaultdict(int)
    
    for emotion_dir in emotion_dirs:
        true_label = normalize_label(emotion_dir.name)
        audio_files = find_audio_files(emotion_dir)
        
        # Apply limit if specified
        if args.limit_per_class is not None:
            audio_files = audio_files[:args.limit_per_class]
        
        for audio_file in audio_files:
            samples.append((audio_file, true_label))
            label_counts[true_label] += 1
    
    if not samples:
        print("Error: No audio files found in the dataset", file=sys.stderr)
        return 1
    
    if not args.quiet:
        print(f"Found {len(samples)} samples across {len(label_counts)} classes")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")
        print()
    
    # Initialize metrics
    class_metrics: dict[str, ClassMetrics] = {
        label: {"true_positives": 0, "false_positives": 0, "false_negatives": 0, "total": 0, "correct": 0}
        for label in set(CANONICAL_LABELS) | set(label_counts.keys())
    }
    
    # Confusion matrix: confusion[true][pred] = count
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    # Track errors
    errors: list[dict] = []
    total_correct = 0
    total_samples = 0
    
    # Run evaluation
    iterator = tqdm(samples, desc="Evaluating", disable=args.quiet)
    for audio_path, true_label in iterator:
        try:
            result = predict_clip(
                path_or_bytes=audio_path,
                model_id=args.model,
                device=args.device,
            )
            pred_label = result.emotion
            
            # Update metrics
            total_samples += 1
            class_metrics[true_label]["total"] += 1
            confusion[true_label][pred_label] += 1
            
            if pred_label == true_label:
                total_correct += 1
                class_metrics[true_label]["correct"] += 1
                class_metrics[true_label]["true_positives"] += 1
            else:
                class_metrics[true_label]["false_negatives"] += 1
                class_metrics[pred_label]["false_positives"] += 1
                
        except (AudioIOError, ModelError) as e:
            errors.append({
                "file": str(audio_path),
                "error": e.message,
                "code": e.code,
            })
            continue
        except Exception as e:
            errors.append({
                "file": str(audio_path),
                "error": str(e),
                "code": "UNKNOWN",
            })
            continue
    
    # Calculate per-class and macro metrics
    per_class_results: dict[str, dict] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    num_classes_with_samples = 0
    
    for label in sorted(class_metrics.keys()):
        metrics = class_metrics[label]
        if metrics["total"] == 0:
            continue
        
        num_classes_with_samples += 1
        accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
        precision, recall, f1 = compute_precision_recall_f1(
            metrics["true_positives"],
            metrics["false_positives"],
            metrics["false_negatives"],
        )
        
        per_class_results[label] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total": metrics["total"],
            "correct": metrics["correct"],
        }
        
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
    
    if num_classes_with_samples > 0:
        macro_precision /= num_classes_with_samples
        macro_recall /= num_classes_with_samples
        macro_f1 /= num_classes_with_samples
    
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Print results
    print("\nEvaluation Results")
    print("==================")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {overall_accuracy:.1%}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Macro Precision: {macro_precision:.3f}")
    print(f"Macro Recall: {macro_recall:.3f}")
    
    if errors:
        print(f"Errors: {len(errors)} files failed to process")
    
    print("\nPer-class results:")
    print("-" * 50)
    for label in sorted(per_class_results.keys()):
        r = per_class_results[label]
        print(f"{label:12s} Acc: {r['accuracy']:5.1%}  F1: {r['f1']:.3f}  ({r['correct']}/{r['total']})")
    
    # Print confusion matrix if requested
    if args.confusion and confusion:
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print("-" * 50)
        
        labels = sorted(set(
            list(confusion.keys()) + 
            [p for c in confusion.values() for p in c.keys()]
        ))
        
        # Header
        header = f"{'':12s}" + "".join(f"{l[:4]:>6s}" for l in labels)
        print(header)
        
        # Rows
        for true_label in labels:
            row = f"{true_label[:12]:12s}"
            for pred_label in labels:
                count = confusion[true_label][pred_label]
                row += f"{count:6d}"
            print(row)
    
    # Save JSON if requested
    if args.json:
        results = {
            "total_samples": total_samples,
            "overall_accuracy": overall_accuracy,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "per_class": per_class_results,
            "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
            "errors": errors,
        }
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
