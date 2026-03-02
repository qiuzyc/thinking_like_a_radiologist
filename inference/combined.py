import argparse
import torch
from inference import main_inference
from detokenization import main_detoken


def main():
    parser = argparse.ArgumentParser(description="Run inference and detokenization pipeline")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--inference_jsonl_path", type=str, required=True, help="Path to input JSONL file for inference")
    parser.add_argument("--inference_output_path", type=str, required=True, help="Path to save inference output")
    parser.add_argument("--detoken_output_path", type=str, required=True, help="Path to save detokenized output")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA weights")
    args = parser.parse_args()

    # Run inference
    main_inference(
        args.base_model_path,
        args.inference_jsonl_path,
        args.inference_output_path
    )

    main_detoken(
        args.base_model_path,
        args.inference_output_path,
        args.detoken_output_path,
        args.lora_path
    )

if __name__ == "__main__":
    main()