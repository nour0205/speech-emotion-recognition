#!/usr/bin/env python3
"""Generate emotion timeline from an audio file.

This script runs timeline-based emotion recognition on an audio file,
applying windowing, inference, smoothing, and segment merging to produce
a comprehensive emotion timeline.

Usage:
    python scripts/predict_timeline.py --input path/to/audio.wav
    python scripts/predict_timeline.py --input audio.wav --window_sec 2.0 --hop_sec 0.5
    python scripts/predict_timeline.py --input audio.wav --smoothing hysteresis --hysteresis_min_run 3
    python scripts/predict_timeline.py --input audio.wav --include_windows --include_scores

Example output:
    {
        "model_name": "baseline",
        "sample_rate": 16000,
        "duration_sec": 10.5,
        "window_sec": 2.0,
        "hop_sec": 0.5,
        "pad_mode": "zero",
        "smoothing": {"method": "hysteresis", "hysteresis_min_run": 3},
        "is_padded_timeline": true,
        "segments": [
            {"start_sec": 0.0, "end_sec": 3.5, "emotion": "neutral", "confidence": 0.82},
            {"start_sec": 3.5, "end_sec": 7.0, "emotion": "happy", "confidence": 0.91},
            {"start_sec": 7.0, "end_sec": 10.5, "emotion": "neutral", "confidence": 0.78}
        ]
    }
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from audioio import AudioConfig
from audioio.errors import AudioIOError
from model.errors import ModelError
from timeline import (
    MergeConfig,
    SmoothingConfig,
    WindowingConfig,
    generate_timeline,
)
from timeline.errors import TimelineError


def main() -> int:
    """Main entry point for the script.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        description="Generate emotion timeline from an audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --input speech.wav
    %(prog)s --input speech.wav --window_sec 3.0 --hop_sec 1.0
    %(prog)s --input speech.wav --smoothing majority --majority_window 5
    %(prog)s --input speech.wav --smoothing hysteresis --hysteresis_min_run 4
    %(prog)s --input speech.wav --smoothing ema --ema_alpha 0.7
    %(prog)s --input speech.wav --include_windows --include_scores --pretty
        """,
    )
    
    # Input/output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input audio file (WAV format)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print JSON output with indentation",
    )
    
    # Model arguments
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
    
    # Windowing arguments
    parser.add_argument(
        "--window_sec",
        type=float,
        default=2.0,
        help="Window duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--hop_sec",
        type=float,
        default=0.5,
        help="Hop/stride duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--pad_mode",
        type=str,
        choices=["none", "zero", "reflect"],
        default="zero",
        help="Padding mode for partial windows (default: zero)",
    )
    
    # Smoothing arguments
    parser.add_argument(
        "--smoothing",
        type=str,
        choices=["none", "majority", "hysteresis", "ema"],
        default="hysteresis",
        help="Smoothing method (default: hysteresis)",
    )
    parser.add_argument(
        "--majority_window",
        type=int,
        default=5,
        help="Window size for majority voting (must be odd, default: 5)",
    )
    parser.add_argument(
        "--hysteresis_min_run",
        type=int,
        default=3,
        help="Minimum consecutive windows to switch emotion (default: 3)",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.6,
        help="EMA alpha coefficient (default: 0.6)",
    )
    
    # Merge arguments
    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="Disable merging of adjacent segments with same emotion",
    )
    parser.add_argument(
        "--min_segment_sec",
        type=float,
        default=0.25,
        help="Minimum segment duration in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--drop_short_segments",
        action="store_true",
        help="Merge segments shorter than min_segment_sec into neighbors",
    )
    parser.add_argument(
        "--short_segment_strategy",
        type=str,
        choices=["merge_prev", "merge_next", "merge_best"],
        default="merge_best",
        help="How to handle short segments (default: merge_best)",
    )
    
    # Output control arguments
    parser.add_argument(
        "--include_windows",
        action="store_true",
        help="Include per-window predictions in output",
    )
    parser.add_argument(
        "--include_scores",
        action="store_true",
        help="Include per-label scores in segments and windows",
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(json.dumps({
            "error": "File not found",
            "code": "FILE_NOT_FOUND",
            "path": str(input_path),
        }), file=sys.stderr)
        return 1
    
    try:
        # Build configs
        windowing_config = WindowingConfig(
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            pad_mode=args.pad_mode,
        )
        
        smoothing_config = SmoothingConfig(
            method=args.smoothing,
            majority_window=args.majority_window,
            hysteresis_min_run=args.hysteresis_min_run,
            ema_alpha=args.ema_alpha,
        )
        
        merge_config = MergeConfig(
            merge_adjacent=not args.no_merge,
            min_segment_sec=args.min_segment_sec,
            drop_short_segments=args.drop_short_segments,
            short_segment_strategy=args.short_segment_strategy,
        )
        
        # Generate timeline
        result = generate_timeline(
            path_or_bytes=input_path,
            audio_config=AudioConfig(),
            windowing_config=windowing_config,
            model_id=args.model,
            device=args.device,
            smoothing_config=smoothing_config,
            merge_config=merge_config,
            include_windows=args.include_windows,
            include_scores=args.include_scores,
        )
        
        # Convert to dict for JSON output
        output = result.to_dict(
            include_windows=args.include_windows,
            include_scores=args.include_scores,
        )
        
        # Format output
        if args.pretty:
            json_output = json.dumps(output, indent=2, ensure_ascii=False)
        else:
            json_output = json.dumps(output, ensure_ascii=False)
        
        # Write output
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json_output + "\n")
            print(f"Timeline written to {output_path}", file=sys.stderr)
        else:
            print(json_output)
        
        return 0
        
    except (AudioIOError, ModelError, TimelineError) as e:
        error_output = {
            "error": str(e),
            "code": getattr(e, "code", "UNKNOWN_ERROR"),
            "type": type(e).__name__,
        }
        if hasattr(e, "details") and e.details:
            error_output["details"] = e.details
        print(json.dumps(error_output), file=sys.stderr)
        return 2
    
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "code": "UNEXPECTED_ERROR",
            "type": type(e).__name__,
        }), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
