import torch
import json
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from PIL import Image
from tqdm import tqdm

def load_and_tokenize_prompts(file_path, processor, model):
    """Load and tokenize prompts from JSONL file with progress bar"""
    prompts = []
    responses = []
    tokens_list = []
    
    # 先统计总行数用于进度条
    print("Counting lines in file...")
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Processing {total_lines} lines...")
    
    number = 1

    with open(file_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Tokenizing prompts"):
            data = json.loads(line)
            number += 1
            if "prompt" in data:
                if 'images' in data:
                    images = [images = [Image.open(img_path) for img_path in data['images']]]
                    input_images = images[:1]
                    output_images = images[1:] 

                    if input_images:
                        inputs_prompt = processor(data['prompt'], images=input_images, padding=False, return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)
                    else:
                        inputs_prompt = processor(data['prompt'], padding=False, return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)

                    if output_images:
                        inputs_response = processor(data['response'], images=output_images, padding=False, return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)
                    else:
                        inputs_response = processor(data['response'], padding=False, return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)
                    
                    inputs_prompt_ids = inputs_prompt['input_ids']
                    inputs_response_ids = inputs_response['input_ids']

                    if input_images:
                        pixel_values = inputs_prompt["pixel_values"]
                        image_tokens = model.get_image_tokens(pixel_values)
                        special_image_mask = inputs_prompt_ids == 8711  # Image token ID
                        image_tokens = image_tokens.to(inputs_prompt_ids.device, inputs_prompt_ids.dtype)
                        inputs_prompt_ids = inputs_prompt_ids.masked_scatter(special_image_mask, image_tokens)
                    
                    if output_images:
                        pixel_values = inputs_response["pixel_values"]
                        image_tokens = model.get_image_tokens(pixel_values)
                        special_image_mask = inputs_response_ids == 8711  # Image token ID
                        image_tokens = image_tokens.to(inputs_response_ids.device, inputs_response_ids.dtype)
                        inputs_response_ids = inputs_response_ids.masked_scatter(special_image_mask, image_tokens)
                    
                    input_ids = inputs_prompt_ids[0].tolist() + [8710] + inputs_response_ids[0].tolist()[1:] + [2]

                else:
                    inputs_prompt = processor(data['prompt'], padding=False, return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)
                    inputs_response = processor(data['response'], padding=False, return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)
                    
                    inputs_prompt_ids = inputs_prompt['input_ids']
                    inputs_response_ids = inputs_response['input_ids']
                    
                    input_ids = inputs_prompt_ids[0].tolist() + [8710] + inputs_response_ids[0].tolist()[1:] + [2]

                prompts.append(data['prompt'])
                responses.append(data['response'])
                tokens_list.append(input_ids)
    
    print(f"Finished processing. Loaded {len(prompts)} prompts.")
    return prompts, responses, tokens_list


def save_tokenized_to_jsonl(prompts, responses, tokens_list, output_path):
    """Save tokenized prompts to JSONL file"""
    with open(output_path, 'w') as f:
        for prompt, response, tokens in zip(prompts, responses, tokens_list):
            output = {
                "prompt": prompt,
                "response": response, 
                "tokens": tokens,
            }
            json.dump(output, f)
            f.write('\n')


# Example usage:
def tokenize_jsonl_file(input_file, output_file, model_path):
    """
    Main function to tokenize a JSONL file of prompts
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file with tokenized prompts
        model_path: Path to the model (required for processor)
    """
    # Load model and processor
    processor = ChameleonProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    model = ChameleonForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Load and tokenize prompts
    prompts, responses, tokens_list = load_and_tokenize_prompts(
        input_file, processor, model.model
    )
    
    # Save tokenized results
    save_tokenized_to_jsonl(
        prompts, responses, tokens_list, output_file
    )
    
    print(f"Tokenized {len(prompts)} prompts and saved to {output_file}")

tokenize_jsonl_file("your/path/to/input/jsonl", "your/path/to/tokenized/jsonl", model_path="../Anole-zebra-cot")