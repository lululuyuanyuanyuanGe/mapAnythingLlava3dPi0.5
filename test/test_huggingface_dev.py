import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

parser = argparse.ArgumentParser("Huggingface AutoModel Tesing")
parser.add_argument("--model_name_or_path", default="", help="pretrained model name or path.")
parser.add_argument("--num_images", type=int, default=1, help="num_images for testing.")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device: cuda or cpu")
parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16","float32"], help="torch dtype")

args = parser.parse_args()
if __name__ == "__main__":
    model_name_or_path = Path(args.model_name_or_path)
    
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    print(processor.statistics)

    # map dtype and set device
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype).eval()
    # Ensure LLaVA-3D decoding path is enabled for replaced language model
    if hasattr(model, "config"):
        setattr(model.config, "use_llava3d", True)
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    image = Image.open("test/example.png").convert("RGB")
    images = [image] * args.num_images
    prompt = "What action should the robot take to pick the cup?"
    inputs = processor(images=images, text=prompt, unnorm_key="bridge_orig/1.0.0", return_tensors="pt")
    # Inject first-step cache position to ensure pixel_values are included in generation inputs
    inputs["cache_position"] = torch.tensor([0], dtype=torch.long)
    # Move inputs to the correct device
    if args.device == "cuda" and torch.cuda.is_available():
        inputs = inputs.to(device=model.device)
    print(inputs)
    generation_outputs = model.predict_action(inputs)
    print(generation_outputs, processor.batch_decode(generation_outputs))

    actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")
    print(actions)
    
    print("DONE!")