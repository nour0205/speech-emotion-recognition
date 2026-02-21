#!/usr/bin/env python3
"""Predict emotion from a single audio file.

This script runs emotion inference on a single audio file and outputs
the result as JSON to stdout.

Usage:
    python scripts/predict_file.py --input path/to/audio.wav
    python scripts/predict_file.py --input path/to/audio.wav --model baseline
    python scripts/predict_file.py --input path/to/audio.wav --device cuda

Example output:
    {
        "emotion": "happy",
        "confidence": 0.85,
        "scores": {"neutral": 0.05, "happy": 0.85, "sad": 0.05, "angry": 0.05, ...},
        "model_name": "speechbrain-iemocap",
        "raw_label": "hap",
        "raw_scores": {"ang": 0.05, "hap": 0.85, "neu": 0.05, "sad": 0.05},
        "duration_sec": 2.5
    }
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model import predict_clip
from model.errors import ModelError
from audioio.errors import AudioIOError


def main() -> int:
    """Main entry point for the script.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        description="Predict emotion from an audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --input speech.wav
    %(prog)s --input speech.wav --model baseline
    %(prog)s --input speech.wav --device cuda
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input audio file (WAV format)",
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
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print JSON output with indentation",
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
        # Run inference
        result = predict_clip(
            path_or_bytes=input_path,
            model_id=args.model,
            device=args.device,
        )
        
        # Output result
        output = result.to_dict()
        if args.pretty:
            print(json.dumps(output, indent=2))
        else:
            print(json.dumps(output))
        
        return 0
        
    except AudioIOError as e:
        print(json.dumps({
            "error": e.message,
            "code": e.code,
            "details": e.details,
        }), file=sys.stderr)
        return 2
        
    except ModelError as e:
        print(json.dumps({
            "error": e.message,
            "code": e.code,
            "details": e.details,
        }), file=sys.stderr)
        return 3
        
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "code": "UNKNOWN_ERROR",
        }), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
