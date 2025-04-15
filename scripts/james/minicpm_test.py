from email.mime import image
from src.models.minicpm import MiniCPMV26

import os
from transformers import AutoTokenizer
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torchvision.transforms.v2
from src.models.minicpm import preprocess_for_attack

import matplotlib.pyplot as plt

def visualize_patch(patch_tensor: torch.Tensor, filename: str = "patch_vis.png", out_dir: str = "."):
    """
    Visualize patches stored in shape [3, patch_size, N_patches], where each patch is
    flattened as 14x14=196 split across channels.

    Args:
        patch_tensor (torch.Tensor): shape [3, patch_size, N_patches]
        filename (str): Output filename
        out_dir (str): Directory to save
    """
    os.makedirs(out_dir, exist_ok=True)

    C, patch_size, N_patches = patch_tensor.shape
    patch_area = patch_size * patch_size

    # Reshape into flattened 196-length patches per channel
    try:
        unfolded = patch_tensor.view(C, patch_size, -1, patch_size)  # [3, 14, N, 14]
        patches = unfolded.permute(2, 0, 1, 3)  # [N, C, 14, 14]
    except Exception as e:
        print(f"Error reshaping patches: {e}")
        return

    # Determine grid size
    num_cols = int(N_patches**0.5)
    num_rows = (N_patches + num_cols - 1) // num_cols

    # Make a canvas
    grid = torch.ones(3, num_rows * patch_size, num_cols * patch_size)

    for idx, patch in enumerate(patches):
        row = idx // num_cols
        col = idx % num_cols
        grid[:, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = patch

    grid_img = grid.permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid_img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f">> Patch visualization saved to {out_path}")

os.environ["HF_HOME"] = "/workspace/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/huggingface_cache/hub"

torch.manual_seed(1234)
IMAGE_SIZE=448
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model: MiniCPMV26 = MiniCPMV26().to(device)
#model.disable_model_gradients()

image_path = "images/trina/000.jpg"
pil_image = Image.open(image_path, mode="r")
width, height = pil_image.size
max_dim = max(width, height)
pad_width = (max_dim - width) // 2
pad_height = (max_dim - height) // 2
transform_pil_image = torchvision.transforms.v2.Compose(
    [
        torchvision.transforms.v2.Pad(
            (pad_width, pad_height, pad_width, pad_height), fill=0
        ),
        torchvision.transforms.v2.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        torchvision.transforms.v2.ToTensor(),  # This divides by 255.
    ]
)
image: torch.Tensor = transform_pil_image(pil_image).unsqueeze(0)

print(f"Transformed to image: {image}")


response = model.generate(image=image.squeeze(0), prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"])
print(response)



batch = model.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
    prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"],
    targets=["L"],
)

print("BATCH")
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
print(batch["labels"].shape)
print(batch["image_bound"])
print(len(batch["pixel_values"]))
print(batch["pixel_values"][0][0].shape)

loss = model.compute_loss(
    image=image.squeeze(0),
    input_ids=batch["input_ids"].to(device=device),
    attention_mask=batch["attention_mask"].to(device=device),
    labels=batch["labels"].to(device=device),
    image_bound=batch["image_bound"]
)
print(f"Loss: {loss.item()}")

import pdb
#pdb.set_trace()



model_generations = []

if model.model.processor is None:
    model.model.processor = AutoProcessor.from_pretrained(model.model.config._name_or_path, trust_remote_code=True)
processor = model.model.processor
max_inp_length=8192
max_slice_nums=None
use_image_id=None
system_prompt=''
prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"]
for i, prompt in enumerate(prompts):
# Wrap prompts into chat-style message dicts
    msgs_list = []
    content_parts = ["(<image>./</image>)", prompt]
    user_msg = {"role": "user", "content": "\n".join(content_parts)}

    # System prompt (if any)
    full_msg = []
    if system_prompt:
        full_msg.append({"role": "system", "content": system_prompt})
    full_msg.append(user_msg)

    msgs_list.append(full_msg)
    prompts_str = [
        processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in msgs_list
    ]
    # Same image for each input
    images = [[image.squeeze(0)]]

    # Call processor
    inputs = processor(
        prompts_str,
        images,
        return_tensors="pt",
        max_length=max_inp_length,
        max_slice_nums=max_slice_nums,
        use_image_id=use_image_id,
    )
    print("GEN2")
    print(inputs["input_ids"].shape)
    print(inputs["attention_mask"].shape)
    print(inputs["image_bound"])
    print(len(inputs["pixel_values"]))
    print(inputs["pixel_values"][0][0].shape)
    image_preprocesses = preprocess_for_attack([[image.squeeze(0)]])

    inputs["pixel_values"] = image_preprocesses["pixel_values"]
    inputs["tgt_sizes"] = image_preprocesses["tgt_sizes"]
    pdb.set_trace()
    inputs.pop("image_sizes")
    inputs.to(model.model.device)
    res = model.model.generate(
        **inputs,
        tokenizer=model.tokenizer,
        vision_hidden_states=None,
        stream=False,
        decode_text=True,
        do_sample=True if model.generation_kwargs["temperature"] > 0 else False,
        **model.generation_kwargs
    )
    model_generations.append(res[0])
    print(model_generations)


    print("original mean/std:", inputs["pixel_values"][0][0].mean().item(), inputs["pixel_values"][0][0].std().item())
    print("differentiable mean/std:", batch["pixel_values"][0][0].mean().item(), batch["pixel_values"][0][0].std().item())
    
    visualize_patch(inputs["pixel_values"][0][0])
    visualize_patch(batch["pixel_values"][0][0], "patch_vis2.png")


