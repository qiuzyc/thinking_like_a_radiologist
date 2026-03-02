import os
import torch
import argparse
import json
from PIL import Image
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from transformers.image_transforms import to_pil_image
from typing import List, Dict, Any
from tqdm import tqdm


def split_tokens(tokens: List[int], boi: int, eoi: int) -> List[tuple]:
    segments = []
    current_segment = []
    in_image = False
    
    for token in tokens:
        if token == boi:
            if current_segment:
                segments.append(("text", current_segment))
                current_segment = []
            in_image = True
        elif token == eoi and in_image:
            segments.append(("image", current_segment))
            current_segment = []
            in_image = False
        else:
            current_segment.append(token)

    if current_segment:
        segment_type = "image" if in_image else "text"
        segments.append((segment_type, current_segment))
    
    return segments


def decode_sample(tokens: List[int], processor, model, output_dir: str, sample_id: str) -> Dict[str, Any]:
    device = next(model.parameters()).device

    segments = split_tokens(tokens, model.vocabulary_mapping.boi_token_id, model.vocabulary_mapping.eoi_token_id)
    
    result_text = ""
    image_paths = []
    image_count = 0
    
    for seg_type, seg_tokens in segments:
        if seg_type == "text":
            # Decode text
            token_tensor = torch.tensor([seg_tokens], device=device, dtype=torch.long)
            text = processor.batch_decode(token_tensor, skip_special_tokens=True)[0]
            result_text += text
            
        elif seg_type == "image":
            if len(seg_tokens) != 1024:
                print(f"Warning: Image has {len(seg_tokens)} tokens, expected 1024")
                result_text += f"<invalid_image_{image_count}>"
                image_paths.append(None)
            else:
                try:
                    token_tensor = torch.tensor([seg_tokens], device=device, dtype=torch.long)
                    pixel_values = model.decode_image_tokens(token_tensor)
                    images = processor.postprocess_pixel_values(pixel_values)
                    image = to_pil_image(images[0].detach().cpu())

                    os.makedirs(os.path.join(output_dir, sample_id), exist_ok=True)
                    image_path = os.path.join(output_dir, sample_id, f"image_{image_count}.png")
                    image.save(image_path)
                    
                    result_text += f"<image_{image_count}>"
                    image_paths.append(image_path)
                    print(f"Saved: {image_path}")
                    
                except Exception as e:
                    print(f"Error decoding image {image_count}: {e}")
                    result_text += f"<error_image_{image_count}>"
                    image_paths.append(None)
            
            image_count += 1
    
    return {
        "sample_id": sample_id,
        "text": result_text,
        "image_paths": image_paths
    }


def process_jsonl(jsonl_path: str, model_path: str, output_dir: str, device: str = "cuda"):

    print("Loading model...")
    processor = ChameleonProcessor.from_pretrained(model_path)
    model = ChameleonForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    
    os.makedirs(output_dir, exist_ok=True)

    results = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Processing samples")):
            try:
                sample = json.loads(line)
                tokens = sample.get('response', [])
                sample_id = sample.get('id', str(i))
                
                if not tokens:
                    print(f"Warning: No tokens in sample {i}")
                    continue
                
                result = decode_sample(tokens, processor, model, output_dir, str(sample_id))
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

    output_file = os.path.join(output_dir, "path/to/file")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Processed {len(results)} samples. Results saved to {output_file}")
    return results


def main_detoken(model_path, jsonl_path, output_path):
    parser = argparse.ArgumentParser(description="Simple token decoder for JSONL files")
    parser.add_argument("--model_path", type=str,default=model_path)
    parser.add_argument("--jsonl_path", type=str, default=jsonl_path)
    parser.add_argument("--output_dir", type=str, default=output_path)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    process_jsonl(args.jsonl_path, args.model_path, args.output_dir, args.device)