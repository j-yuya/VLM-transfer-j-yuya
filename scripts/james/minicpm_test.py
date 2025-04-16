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
from transformers.utils import TensorType
from transformers import AutoProcessor
import matplotlib.pyplot as plt
from torchvision import transforms


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
image_size = 448

transform = transforms.Compose(
    [
        torchvision.transforms.v2.Pad(
            (pad_width, pad_height, pad_width, pad_height), fill=0
        ),
        transforms.Resize(
            (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
        ),
    ]
)
image: torch.Tensor = transform(pil_image)

processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
do_pad = True
max_slice_nums = None
return_tensors = TensorType.PYTORCH
processed_image =processor.image_processor([[image]], do_pad=do_pad, max_slice_nums=max_slice_nums, return_tensors=return_tensors)
pixel_values = processed_image["pixel_values"][0][0]
tgt_sizes = processed_image["tgt_sizes"][0][0]

import numpy as np
# response = model.generate(image=image.squeeze(0), prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"])
# print(response)
def pil_to_tensor_255(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a PyTorch tensor (CHW), values in [0, 255], dtype=torch.uint8.
    """
    np_img = np.array(pil_img)  # shape: [H, W, C], dtype=uint8
    if np_img.ndim == 2:  # grayscale image
        np_img = np_img[:, :, None]
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)  # to CHW
    return tensor  # dtype: torch.uint8
gen_image = pil_to_tensor_255(pil_image)


batch = model.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
    prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"],
    targets=["C"],
)

print("BATCH")
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
print(batch["labels"].shape)
print(batch["image_bound"])
print(len(batch["pixel_values"]))
print(batch["pixel_values"][0][0].shape)

model.tgt_sizes = tgt_sizes
loss = model.compute_loss(
    image=pixel_values,
    input_ids=batch["input_ids"].to(device=device),
    attention_mask=batch["attention_mask"].to(device=device),
    labels=batch["labels"].to(device=device),
    image_bound=batch["image_bound"]
)
print(f"Loss: {loss.item()}")
tensor = pil_to_tensor_255(image)
answer = model.generate(tensor, prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"])
print(answer)
import pdb
#pdb.set_trace()

from torchvision.transforms import ToPILImage
img = ToPILImage()(tensor)
img.save("output.jpg")



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


