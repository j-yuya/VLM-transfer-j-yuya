# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
from src.models.qwen_utils.modeling_qwen import QWenLMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import lightning
import torch
from typing import Any, Callable, Dict, List, Mapping, Optional, Literal, Tuple
from transformers import PreTrainedTokenizer, AutoProcessor
from torch.nn.utils.rnn import pad_sequence
from src.models.base import VisionLanguageModel
from PIL import Image
from transformers.utils import TensorType
from torchvision import transforms

from src.models.qwen_utils.visual import VisionTransformer
LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

# Labels with these indices will be ignored by cross entropy loss in PyTorch.
IGNORE_INDEX = -100


class MiniCPMV26(VisionLanguageModel, lightning.LightningModule):
    def __init__(
        self,
        #TODO: More Param Variants
        model_str: str = "MiniCPM-V-2_6",
        generation_kwargs: Mapping[str, Any] | None = None,
        precision: str = "bf16-mixed",
        image_size: int = 448,
    ):
        super().__init__(image_size)
        self.already_logged_new_mask: bool = False  # For print debugigng
        self.already_logged_text: bool = True  # For print debugigng
        if generation_kwargs is None:
            generation_kwargs = {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_new_tokens": 100,
                "min_new_tokens": 5,
            }

        self.model_str = model_str
        self.generation_kwargs = generation_kwargs

        self.precision_str = precision
        if self.precision_str in {"bf16-mixed", "bf16-true"}:
            self.precision_dtype = torch.bfloat16
        elif self.precision_str == "16-true":
            self.precision_dtype = torch.float16
        elif self.precision_str in {"32", "32-true"}:
            self.precision_dtype = torch.float32
        elif self.precision_str in {"64", "64-true"}:
            self.precision_dtype = torch.float64
        else:
            raise ValueError(f"Invalid precision: {self.precision_str}")

        model_path = f"openbmb/{model_str}"

        # not sure why we need to register the image processor manually
        print(f"Using MiniCPMV2.6 model: {model_path}")

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # type: ignore
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.precision_dtype,
            trust_remote_code=True).eval().cuda()
        
        self.tgt_sizes=None
        self.image_sizes=None
        
        #self.already_logged_new_mask: bool = False  # For print debugigng
        #self.already_logged_text: bool = False  # For print debugigng

    def create_images_transform_fn(self, model_str: str) -> Callable:
        raise NotImplementedError(
            "create_images_transform_fn is not implemented for DeepSeek models."
        )

    # TODO: Needs implementation
    def compute_loss(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        image_bound: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        device = self.model.device
        if self.model.processor is None:
            self.model.processor = AutoProcessor.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)
        processor = self.model.processor
        
        do_pad = True
        max_slice_nums:int = None
        return_tensors = TensorType.PYTORCH
        # Verify input image shape
        if image.ndim == 4:
            image = image.squeeze(0)
        print(image)
        assert image.ndim == 3, f"Expected 3D image tensor, got {image.ndim}"
        #assert image.size(0) == 1, f"Expected a single image (B=1), got {image.size(0)}"

        # Repeat image for each batch sample and wrap in list for multimodal input
        B = input_ids.shape[0]

        # If you only have a single image and want to replicate it for each item in the batch:
        images = [[image] for _ in range(B)]
        tgt_sizes = [self.tgt_sizes.unsqueeze(0) for _ in range(B)]
        print(images)
        print(tgt_sizes)
        #print(scaled_image)
        #repeat(B, 1, 1, 1)  # [B, C, H, W]

        #images = [[img] for img in repeated_image]
        # image_inputs = preprocess_for_attack([[image]])
        # images, image_sizes, tgt_sizes = image_inputs["pixel_values"], image_inputs["image_sizes"], image_inputs["tgt_sizes"]


        inputs = MiniCPMVBatchFeature(data={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": images,
            "image_bound": image_bound,
            "tgt_sizes": tgt_sizes
        })
        # print(input_ids)
        # print(attention_mask)
        # print(images)
        # print(image_sizes)
        # print(image_bound)
        # print(tgt_sizes)
        position_ids = (inputs["attention_mask"].cumsum(dim=1) - 1).clamp(min=0)
        inputs["position_ids"] = position_ids
        outputs = self.model(
            data=inputs.to(device),
            labels=labels.to(device)
        )
        return outputs.loss if hasattr(outputs, "loss") else outputs[0]


    def convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        assert self.pad_token_id is not None, "Expected pad token id to be set."
        assert targets is not None, "Not support yet."
        placeholder_image = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
        
        input_by_model = self.build_inputs_with_single_image(prompts, targets, placeholder_image)
        input_by_model["input_ids"] = input_by_model["input_ids"].long()
        input_by_model["attention_mask"] = input_by_model["attention_mask"].long()
        if self.tgt_sizes is None:
            self.tgt_sizes=input_by_model["tgt_sizes"][0][0]
        if self.image_sizes is None:
            self.image_sizes=input_by_model["image_sizes"][0][0]
        if targets[0] is not None:
            labels = make_labels(
                input_ids=input_by_model["input_ids"],
                pad_token_id=self.pad_token_id,
                targets=targets,
                tokenizer=self.tokenizer,
            )
            
        results = input_by_model
        results.pop("image_sizes")
        results["labels"] = labels
        print(input_by_model.keys())
        #print(input_by_model["image_sizes"][0])
        if not self.already_logged_text:
            torch.set_printoptions(threshold=10000)
            # first_text = prompt_texts[0]
            # print(f"First text: {first_text}")
            print(f"First input_ids: {input_by_model['input_ids'][0]}")
            print(f"First attention_mask: {input_by_model['attention_mask'][0]}")
            print(f"First labels: {results['labels'][0]}")
            if len(input_by_model['input_ids']) > 1:
                print(f"Second input ids: {input_by_model['input_ids'][1]}")
                print(f"Second attention_mask: {input_by_model['attention_mask'][1]}")
                print(f"Second labels: {results['labels'][1]}")
            # non_minus_100 = [r for r in results["labels"][0] if r != IGNORE_INDEX]
            # non_minus_100_text = self.tokenizer.decode(non_minus_100)
            # print(f"Example text that we calculate loss on: {non_minus_100_text}")
            torch.set_printoptions(profile="default")
            self.already_logged_text = True

        return results

    # TODO: Needs implementation
    @torch.inference_mode()
    def generate(self, image: torch.Tensor, prompts: List[str]) -> List[str]:
        # We should only have a single image.
        if image.ndim == 4:
            image = image.squeeze(0)
        assert image.ndim == 3, f"Expected (3, H, W), got {image.shape}"
        model_generations = []

        if self.model.processor is None:
            self.model.processor = AutoProcessor.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)
        processor = self.model.processor
        max_inp_length=8192
        max_slice_nums=None
        use_image_id=None
        system_prompt=''
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
            images = [[image]]

            # image_data = {"pixel_values":[[image]], "image_sizes": [[self.image_sizes]], "tgt_sizes": [[self.tgt_sizes]]}
            # inputs = processor._convert_images_texts_to_inputs(image_data, prompts_str, max_slice_nums=max_slice_nums, use_image_id=use_image_id, max_length=max_inp_length)
            inputs = processor(prompts_str, images, max_slice_nums=max_slice_nums, use_image_id=use_image_id, max_length=max_inp_length)
            print("GEN")
            print(inputs["input_ids"].shape)
            print(inputs["attention_mask"].shape)
            print(inputs["image_bound"])
            print(len(inputs["pixel_values"]))
            print(inputs["pixel_values"][0][0].shape)
            print(inputs["tgt_sizes"])
            print(inputs["image_sizes"])
            # import pdb
            # pdb.set_trace()
            inputs.pop("image_sizes")
            print("original mean/std:", inputs["pixel_values"][0][0].mean().item(), inputs["pixel_values"][0][0].std().item())

            inputs.to(self.model.device)
            res = self.model.generate(
                **inputs,
                tokenizer=self.tokenizer,
                vision_hidden_states=None,
                stream=False,
                decode_text=True,
                do_sample=True if self.generation_kwargs["temperature"] > 0 else False,
                **self.generation_kwargs
            )
            model_generations.append(res[0])
        return model_generations
        

    def disable_model_gradients(self):
        """
        Disables gradient computation for all components of MiniCPMV and sets them to eval mode.
        """

        # Disable gradients and set eval mode for full model
        self.model.requires_grad_(False)
        self.model.eval()

        # Explicitly handle submodules
        self.model.llm.requires_grad_(False)
        self.model.llm.eval()

        self.model.llm.model.requires_grad_(False)
        self.model.llm.model.eval()

        self.model.llm.model.embed_tokens.requires_grad_(False)
        self.model.llm.model.norm.requires_grad_(False)
        self.model.llm.lm_head.requires_grad_(False)

        self.model.vpm.requires_grad_(False)
        self.model.vpm.eval()

        self.model.vpm.embeddings.requires_grad_(False)
        self.model.vpm.encoder.requires_grad_(False)
        self.model.vpm.post_layernorm.requires_grad_(False)

        self.model.resampler.requires_grad_(False)
        self.model.resampler.eval()

        self.model.resampler.kv_proj.requires_grad_(False)
        self.model.resampler.attn.requires_grad_(False)
        self.model.resampler.ln_q.requires_grad_(False)
        self.model.resampler.ln_kv.requires_grad_(False)
        self.model.resampler.ln_post.requires_grad_(False)

    #TODO: generation args not correct
    def build_inputs_with_single_image(
        self,
        prompts: List[str],
        targets: Optional[List[str]],
        image: "PIL.Image.Image",
        processor=None,
        system_prompt: str = "",
        max_inp_length=8192,
        max_slice_nums=None,
        use_image_id=None,
    ):
        if processor is None:
            if self.model.processor is None:
                self.model.processor = AutoProcessor.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)
            processor = self.model.processor

        # Wrap prompts into chat-style message dicts
        msgs_list = []
        for i, prompt in enumerate(prompts):
            content_parts = ["(<image>./</image>)", prompt]
            user_msg = {"role": "user", "content": "\n".join(content_parts)}
            assistant_msg = {"role": "assistant", "content": targets[i]} if targets is not None else None

            # System prompt (if any)
            full_msg = []
            if system_prompt:
                full_msg.append({"role": "system", "content": system_prompt})
            full_msg.append(user_msg)
            if assistant_msg:
                full_msg.append(assistant_msg)

            msgs_list.append(full_msg)

        # Convert to prompt strings

        prompts_str = [
            processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=targets is None)
            for msgs in msgs_list
        ]

        # Same image for each input
        images = [[image] for _ in range(len(prompts))]

        # Call processor
        inputs = processor(
            prompts_str,
            images,
            return_tensors="pt",
            max_length=max_inp_length,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
        )

        return inputs


IGNORE_INDEX = -100  # standard for ignoring in loss

def make_labels(
    input_ids: torch.Tensor,
    pad_token_id: int,
    targets: list[str],
    tokenizer
):
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX  # ignore everything by default

    for i, target in enumerate(targets):
        # Build assistant message as it was included in the template
        assistant_prefix = "<|im_start|>assistant\n"
        assistant_suffix = "<|im_end|>"

        full_target_text = assistant_prefix + target + assistant_suffix
        full_target_ids = tokenizer(full_target_text, add_special_tokens=False).input_ids

        # Search for this sequence in the input_ids
        sequence = input_ids[i].tolist()

        for start_idx in range(len(sequence) - len(full_target_ids) + 1):
            if sequence[start_idx:start_idx + len(full_target_ids)] == full_target_ids:
                labels[i, start_idx:start_idx + len(full_target_ids)] = input_ids[i, start_idx:start_idx + len(full_target_ids)]
                break
        else:
            raise ValueError("Target sequence not found in input_ids — template mismatch?")

    # Ignore pad tokens
    labels[input_ids == pad_token_id] = IGNORE_INDEX
    return labels

from typing import Optional, Union, Dict, Any, List

import PIL.Image
import numpy as np
import PIL

from transformers.utils import requires_backends, is_torch_dtype, is_torch_device
from transformers.image_processing_utils import BatchFeature


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)

class MiniCPMVBatchFeature(BatchFeature):
    r"""
    Extend from BatchFeature for supporting various image size
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        if tensor_type is None:
            return self
        
        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        def converter(value):
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    return tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )


        for key, value in self.items():
            self[key] = recursive_converter(converter, value)
        return self
            
    def to(self, *args, **kwargs) -> "MiniCPMVBatchFeature":
        requires_backends(self, ["torch"])
        import torch

        def cast_tensor(v):
            # check if v is a floating point
            if torch.is_floating_point(v):
                # cast and send to device
                return v.to(*args, **kwargs)
            elif device is not None:
                return v.to(device=device)
            else:
                return v

        new_data = {}
        device = kwargs.get("device")
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            new_data[k] = recursive_converter(cast_tensor, v)
        self.data = new_data
        return self
    

import torch
import torch.nn.functional as F
from typing import List, Optional, Union
import math

def preprocess_for_attack(
    images: List[List[torch.Tensor]],
    patch_size: int = 14,
    scale_resolution: int = 448,
    max_slice_nums: int = 9,
    slice_mode: bool = True,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
):
    def normalize(tensor, mean, std):
        mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
        return (tensor - mean) / std

    def ensure_divide(length, patch_size):
        return max(round(length / patch_size) * patch_size, patch_size)

    def find_best_resize(h, w):
        r = w / h
        new_h = int(scale_resolution / (r**0.5))
        new_w = int(new_h * r)
        return ensure_divide(new_h, patch_size), ensure_divide(new_w, patch_size)

    def get_sliced_grid(h, w):
        area = h * w
        
        ratio = h * w / (scale_resolution ** 2)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1 or not slice_mode:
            return None
        best_grid = (1, 1)
        log_ratio = (w / h).log() if isinstance(w, torch.Tensor) else math.log(w / h)
        min_error = float("inf")
        for i in [multiple - 1, multiple, multiple + 1]:
            if i <= 1 or i > max_slice_nums:
                continue
            for rows in range(1, i + 1):
                if i % rows == 0:
                    cols = i // rows
                    err = abs(log_ratio - torch.log(torch.tensor(cols / rows)))
                    if err < min_error:
                        best_grid = (cols, rows)
                        min_error = err
        return best_grid

    def split_tensor_to_patches(tensor, grid):
        C, H, W = tensor.shape
        cols, rows = grid
        patch_h = H // rows
        patch_w = W // cols
        patches = []
        for i in range(rows):
            for j in range(cols):
                patch = tensor[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                patches.append(patch)
        return patches

    def reshape_by_patch(image: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
        """
        Reshape a [C, H, W] image tensor into [C, patch_size, HW // patch_size]
        using unfold, replicating MiniCPMV logic faithfully.

        Args:
            image (torch.Tensor): Tensor of shape [3, H, W]
            patch_size (int): Patch size for unfolding (default: 14)

        Returns:
            torch.Tensor: Tensor of shape [3, patch_size, HW // patch_size]
        """
        assert image.ndim == 3 and image.shape[0] == 3, "Expected image shape [3, H, W]"
        unfolded = F.unfold(image.unsqueeze(0), kernel_size=patch_size, stride=patch_size)  # [1, C*P*P, N]
        C = image.shape[0]
        unfolded = unfolded.view(C, patch_size, patch_size, -1)  # [C, P, P, N]
        reshaped = unfolded.permute(0, 1, 3, 2).reshape(C, patch_size, -1)  # [C, P, P*N] -> [C, P, N*P]
        return reshaped

    all_pixel_values = []
    all_image_sizes = []
    all_tgt_sizes = []

    for img_list in images:
        pixel_values = []
        image_sizes = []
        tgt_sizes = []
        for img in img_list:
            C, H, W = img.shape
            image_sizes.append((W, H))
            grid = get_sliced_grid(H, W)

            if grid is None:
                new_h, new_w = find_best_resize(H, W)
                resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bicubic', align_corners=False).squeeze(0)
                patches = [resized]
            else:
                # resize to grid-compatible size
                new_h = ensure_divide(H, grid[1])
                new_w = ensure_divide(W, grid[0])
                resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bicubic', align_corners=False).squeeze(0)
                patches = split_tensor_to_patches(resized, grid)

            # ✅ Compute tgt_size once based on resized image BEFORE reshape
            H_patches = resized.shape[1] // patch_size
            W_patches = resized.shape[2] // patch_size
            tgt_size = torch.tensor((H_patches, W_patches), device=resized.device)

            patches = [normalize(p, mean, std) for p in patches]
            reshaped = [reshape_by_patch(p) for p in patches]
            tgt_sizes.extend([tgt_size] * len(reshaped))
            pixel_values.extend(reshaped)

        all_pixel_values.append(pixel_values)
        all_image_sizes.append(image_sizes)
        all_tgt_sizes.append(torch.stack(tgt_sizes) if tgt_sizes else torch.empty(0))

    return {
        "pixel_values": all_pixel_values,
        "image_sizes": all_image_sizes,
        "tgt_sizes": all_tgt_sizes,
    }
