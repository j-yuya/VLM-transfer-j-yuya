from email.mime import image
from src.models.cogvlm2 import CogVLM2

import os
from transformers import AutoTokenizer
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torchvision.transforms.v2
from src.models.minicpm import preprocess_for_attack
from transformers.utils import TensorType
from transformers import AutoProcessor
import matplotlib.pyplot as plt
from torchvision import transforms


os.environ["HF_HOME"] = "/workspace/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/huggingface_cache/hub"

torch.manual_seed(1234)
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("THUDM/cogvlm2-llama3-chat-19B", trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model: CogVLM2 = CogVLM2().to(device)
#model.disable_model_gradients()
image_path = "images/trina/000.jpg"
pil_image = Image.open(image_path, mode="r")
width, height = pil_image.size
max_dim = max(width, height)
pad_width = (max_dim - width) // 2
pad_height = (max_dim - height) // 2
image_size = 1344

transform = transforms.Compose(
    [
        torchvision.transforms.v2.Pad(
            (pad_width, pad_height, pad_width, pad_height), fill=0
        ),
        transforms.Resize(
            (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)
image: torch.Tensor = transform(pil_image)


batch = model.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
    prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"],
    targets=["X"],
)

print("BATCH")
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
print(batch["labels"].shape)

loss = model.compute_loss(
    image=image,
    input_ids=batch["input_ids"].to(device=device),
    token_type_ids=batch["token_type_ids"].to(device=device),
    attention_mask=batch["attention_mask"].to(device=device),
    labels=batch["labels"].to(device=device),
)
print(f"Loss: {loss.item()}")



answer = model.generate(image, prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"])
print(answer)

def cog_reverse_image(
    tensor,
    image_size,
) -> Image.Image:
    """
    Reverses the normalization and tensor-to-PIL conversion
    for a 3xHxW image tensor. Assumes CogVLM-style normalization.
    
    Optionally resizes the image to (image_size, image_size).
    """
    # Mean and std from the original transform
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    # Inverse normalization
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]
    inverse_normalize = transforms.Normalize(mean=inv_mean, std=inv_std)

    # Apply inverse normalization
    tensor = inverse_normalize(tensor.clone())  # clone to avoid modifying original

    # Clamp to valid range [0, 1]
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Convert back to a PIL image
    from torchvision.transforms import ToPILImage
    pil_image = ToPILImage()(tensor)

    # Optionally resize to a square if image_size is provided
    if image_size is not None:
        pil_image = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC
        )(pil_image)

    return pil_image


pil_image = cog_reverse_image(image, 448)
pil_image.save("cog_test.jpg")



# model_generations = []

# if model.model.processor is None:
#     model.model.processor = AutoProcessor.from_pretrained(model.model.config._name_or_path, trust_remote_code=True)
# processor = model.model.processor
# max_inp_length=8192
# max_slice_nums=None
# use_image_id=None
# system_prompt=''
# prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"]
# for i, prompt in enumerate(prompts):
# # Wrap prompts into chat-style message dicts
#     msgs_list = []
#     content_parts = ["(<image>./</image>)", prompt]
#     user_msg = {"role": "user", "content": "\n".join(content_parts)}

#     # System prompt (if any)
#     full_msg = []
#     if system_prompt:
#         full_msg.append({"role": "system", "content": system_prompt})
#     full_msg.append(user_msg)

#     msgs_list.append(full_msg)
#     prompts_str = [
#         processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
#         for msgs in msgs_list
#     ]
#     # Same image for each input
#     images = [[image.squeeze(0)]]

#     # Call processor
#     inputs = processor(
#         prompts_str,
#         images,
#         return_tensors="pt",
#         max_length=max_inp_length,
#         max_slice_nums=max_slice_nums,
#         use_image_id=use_image_id,
#     )
#     print("GEN2")
#     print(inputs["input_ids"].shape)
#     print(inputs["attention_mask"].shape)
#     print(inputs["image_bound"])
#     print(len(inputs["pixel_values"]))
#     print(inputs["pixel_values"][0][0].shape)
#     image_preprocesses = preprocess_for_attack([[image.squeeze(0)]])

#     inputs["pixel_values"] = image_preprocesses["pixel_values"]
#     inputs["tgt_sizes"] = image_preprocesses["tgt_sizes"]
#     pdb.set_trace()
#     inputs.pop("image_sizes")
#     inputs.to(model.model.device)
#     res = model.model.generate(
#         **inputs,
#         tokenizer=model.tokenizer,
#         vision_hidden_states=None,
#         stream=False,
#         decode_text=True,
#         do_sample=True if model.generation_kwargs["temperature"] > 0 else False,
#         **model.generation_kwargs
#     )
#     model_generations.append(res[0])
#     print(model_generations)


#     print("original mean/std:", inputs["pixel_values"][0][0].mean().item(), inputs["pixel_values"][0][0].std().item())
#     print("differentiable mean/std:", batch["pixel_values"][0][0].mean().item(), batch["pixel_values"][0][0].std().item())
    
#     visualize_patch(inputs["pixel_values"][0][0])
#     visualize_patch(batch["pixel_values"][0][0], "patch_vis2.png")


