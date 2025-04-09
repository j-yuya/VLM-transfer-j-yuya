from email.mime import image
from src.models.internvl2 import InternVL2

import os
from transformers import AutoTokenizer
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def dynamic_preprocess(image, patch_grid=(2, 2), image_size=448, use_thumbnail=False):
    num_patches_x, num_patches_y = patch_grid
    target_width = image_size * num_patches_x
    target_height = image_size * num_patches_y
    blocks = num_patches_x * num_patches_y

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % num_patches_x) * image_size,
            (i // num_patches_x) * image_size,
            ((i % num_patches_x) + 1) * image_size,
            ((i // num_patches_x) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and blocks != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

def load_image_from_image(image_file, input_size=448, patch_grid=(2, 2), use_thumbnail=True):
    transform = build_transform(input_size=448)
    images = dynamic_preprocess(
        image_file,
        image_size=input_size,
        patch_grid=patch_grid,
        use_thumbnail=use_thumbnail
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


os.environ["HF_HOME"] = "/workspace/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/huggingface_cache/hub"

torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-8B", trust_remote_code=True)
# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model: InternVL2 = InternVL2().to(device)
#model.disable_model_gradients()

image_path = "images/trina/000.jpg"
pil_image = Image.open(image_path, mode="r")

image = load_image_from_image(pil_image)




print(f"Transformed to image: {image}")
#image = transformed_image.unsqueeze(0).to(device)

response = model.generate(image=image.unsqueeze(0), prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"])
print(response)


batch = model.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
    prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"],
    targets=["C"],
)
loss = model.compute_loss(
    image=image.unsqueeze(0),
    input_ids=batch["input_ids"].to(device=device),
    attention_mask=batch["attention_mask"].to(device=device),
    labels=batch["labels"].to(device=device),
)
print(f"Loss: {loss.item()}")
