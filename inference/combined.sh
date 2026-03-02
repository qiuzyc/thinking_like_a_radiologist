BASE_MODEL_PATH="/path/to/your/base/model"
INFERENCE_JSONL_PATH="/path/to/your/input.jsonl"
INFERENCE_OUTPUT_PATH="/path/to/save/inference/output.jsonl"
DETOKEN_OUTPUT_PATH="/path/to/save/detokenized/output.jsonl"
LORA_PATH="/path/to/your/lora/weights"

python combined.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --inference_jsonl_path "$INFERENCE_JSONL_PATH" \
    --inference_output_path "$INFERENCE_OUTPUT_PATH" \
    --detoken_output_path "$DETOKEN_OUTPUT_PATH" \
    --lora_path "$LORA_PATH"