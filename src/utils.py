from accelerate import Accelerator
import ast
import getpass
import joblib
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import torch
import torch.distributed
import torch.utils.data
import torchvision.transforms.v2
from typing import Any, Dict, List, Tuple
import wandb
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import transforms
from src.data import VLMEnsembleTextDataset, VLMEnsembleTextDataModule
from src.models.ensemble import VLMEnsemble
from src.image_handling import get_list_image
from transformers import AutoProcessor
from transformers.utils import TensorType
from torchvision.transforms.functional import to_pil_image
from typing import Optional
from PIL.Image import Image as PILImage 

def calc_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def create_initial_image(image_kwargs: Dict[str, Any], seed: int = 0) -> torch.Tensor:
    if image_kwargs["image_initialization"] == "NIPS17":
        image = get_list_image("old/how_robust_is_bard/src/dataset/NIPS17")
        # resizer = transforms.Resize((224, 224))
        # images = torch.stack(
        #     [resizer(i).unsqueeze(0).to(torch.float16) for i in images]
        # )
        # # Only use one image for one attack.
        # images: torch.Tensor = images[image_kwargs["datum_index"]].unsqueeze(0)
        raise NotImplementedError
    elif image_kwargs["image_initialization"] == "random":
        image_size = image_kwargs["image_size"]
        image: torch.Tensor = torch.rand((1, 3, image_size, image_size))
    elif image_kwargs["image_initialization"] == "trina":
        image_path = f"images/trina/{str(seed).zfill(3)}.jpg"
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
                    (image_kwargs["image_size"], image_kwargs["image_size"])
                ),
                torchvision.transforms.v2.ToTensor(),  # This divides by 255.
            ]
        )
        image: torch.Tensor = transform_pil_image(pil_image).unsqueeze(0)
    else:
        raise ValueError(
            "Invalid image_initialization: {}".format(
                image_kwargs["image_initialization_str"]
            )
        )
    assert len(image.shape) == 4
    return image

def create_intern_image(image_kwargs: Dict[str, Any], seed: int = 0) -> torch.Tensor:
    if image_kwargs["image_initialization"] == "trina":
        image_path = f"images/trina/{str(seed).zfill(3)}.jpg"
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
                    (image_kwargs["image_size"], image_kwargs["image_size"])
                ),
            ]
        )
        image = transform_pil_image(pil_image)
        image = load_image_from_image(image, image_kwargs["image_size"], (1,1), True).unsqueeze(0)
        print(image.shape)
        assert len(image.shape) == 5
        return image
    elif image_kwargs["image_initialization"] == "random":
        image_size = image_kwargs["image_size"]
        image: torch.Tensor = torch.rand((1, 3, image_size, image_size))
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
                    (image_kwargs["image_size"], image_kwargs["image_size"])
                ),
            ]
        )
        image = transform_pil_image(pil_image)
        image = load_image_from_image(image, image_kwargs["image_size"], (1,1), True).unsqueeze(0)
        print(image.shape)
        assert len(image.shape) == 5
        return image
    
def create_cog_image(image_kwargs: Dict[str, Any], seed: int = 0) -> torch.Tensor:
    if image_kwargs["image_initialization"] == "NIPS17":
        image = get_list_image("old/how_robust_is_bard/src/dataset/NIPS17")
        # resizer = transforms.Resize((224, 224))
        # images = torch.stack(
        #     [resizer(i).unsqueeze(0).to(torch.float16) for i in images]
        # )
        # # Only use one image for one attack.
        # images: torch.Tensor = images[image_kwargs["datum_index"]].unsqueeze(0)
        raise NotImplementedError
    elif image_kwargs["image_initialization"] == "random":
        image_size = 1344
        image: torch.Tensor = torch.rand((3, image_size, image_size))
    elif image_kwargs["image_initialization"] == "trina":
        image_path = f"images/trina/{str(seed).zfill(3)}.jpg"
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
    else:
        raise ValueError(
            "Invalid image_initialization: {}".format(
                image_kwargs["image_initialization_str"]
            )
        )
    assert len(image.shape) == 3
    return image

def create_minicpm_image(image_kwargs: Dict[str, Any], seed: int = 0) -> torch.Tensor:
    if image_kwargs["image_initialization"] == "NIPS17":
        image = get_list_image("old/how_robust_is_bard/src/dataset/NIPS17")
        # resizer = transforms.Resize((224, 224))
        # images = torch.stack(
        #     [resizer(i).unsqueeze(0).to(torch.float16) for i in images]
        # )
        # # Only use one image for one attack.
        # images: torch.Tensor = images[image_kwargs["datum_index"]].unsqueeze(0)
        raise NotImplementedError
    elif image_kwargs["image_initialization"] == "random":
        image_size = image_kwargs["image_size"]
        image: torch.Tensor = torch.rand((3, image_size, image_size))
        from torchvision.transforms import ToPILImage
        image = ToPILImage()(image)
        processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
        do_pad = True
        max_slice_nums = None
        return_tensors = TensorType.PYTORCH
        processed_image =processor.image_processor([[image]], do_pad=do_pad, max_slice_nums=max_slice_nums, return_tensors=return_tensors)
        pixel_values = processed_image["pixel_values"][0][0]
        tgt_sizes = processed_image["tgt_sizes"][0][0]
    elif image_kwargs["image_initialization"] == "trina":
        image_path = f"images/trina/{str(seed).zfill(3)}.jpg"
        pil_image = Image.open(image_path, mode="r")
        width, height = pil_image.size
        max_dim = max(width, height)
        pad_width = (max_dim - width) // 2
        pad_height = (max_dim - height) // 2
        image_size = image_kwargs["image_size"]
        
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
    else:
        raise ValueError(
            "Invalid image_initialization: {}".format(
                image_kwargs["image_initialization_str"]
            )
        )
    assert len(pixel_values.shape) == 3
    return pixel_values, tgt_sizes

from typing import Tuple

def reconstruct_cpm_image(
    patch_tensor: torch.Tensor,
    tgt_size: torch.Tensor,  # [H_patches, W_patches]
) -> torch.Tensor:
    """
    Reconstructs image from [3, patch_size, N_patches] using F.fold.
    """
    print("IMAGER")
    
    mean = torch.tensor([0.5, 0.5, 0.5], device=patch_tensor.device, dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], device=patch_tensor.device, dtype=torch.float32)

    img = inverse_reshape_and_unnormalize(patch_tensor, tgt_size, mean, std)
    # print(patch_tensor.shape)
    # print(patch_tensor)
    # print(tgt_size.shape)
    # print(tgt_size)
    # print(mean)
    # print(std)
    # print(mean.shape)
    # print(std.shape)
    # print((std[:, None, None] + mean[:, None, None]).shape)
    # print(img.shape)  # ✅ [3, 448, 448]

    return img


def inverse_reshape_and_unnormalize(
    patch_tensor: torch.Tensor,
    tgt_size: torch.Tensor,  # e.g. [32,32]
    mean: torch.Tensor = None,
    std: torch.Tensor  = None,
    patch_size: int    = 14,
):
    """
    Invert the custom reshape_by_patch(...) that ends up with shape [3,14,14*1024].
    That function does an extra permute(0,1,3,2), so we must 'un-permute'.
    """

    # Convert tgt_size => list
    if isinstance(tgt_size, torch.Tensor):
        tgt_size = tgt_size.tolist()
    h_patches, w_patches = tgt_size  # [32,32]
    num_patches = h_patches * w_patches  # 1024

    # Step A: The final shape from preprocess is [3,14, 14*1024] => [3,14,14336].
    # We must first "unflatten" it to [3,14,1024,14].
    x = patch_tensor.reshape(3, patch_size, num_patches, patch_size)
    # => [3,14,1024,14]

    # Step B: Undo the permute(0,1,3,2). 
    # Swapping dims 2 and 3 is its own inverse, so we do the same permute again:
    x = x.permute(0, 1, 3, 2)  # => [3,14,14,1024]

    # Step C: Flatten to [3*14*14, 1024] => [588, 1024]
    x = x.reshape(3 * patch_size * patch_size, num_patches)

    # Step D: Insert a batch dim => [1,588,1024]
    x = x.unsqueeze(0)

    # Step E: fold => [1,3,448,448]
    H = patch_size * h_patches  # => 448
    W = patch_size * w_patches  # => 448
    x = F.fold(
        x,
        output_size=(H, W),
        kernel_size=(patch_size, patch_size),
        stride=(patch_size, patch_size),
    )
    # => [1,3,448,448]

    # Step F: Remove batch => [3,448,448]
    x = x.squeeze(0)

    # Step G: Unnormalize if requested
    if mean is not None and std is not None:
        mean = mean.to(device=x.device, dtype=x.dtype)[:, None, None]
        std  =  std.to(device=x.device, dtype=x.dtype)[:, None, None]
        x = x * std + mean

    return x.clamp(0, 1)
def dynamic_preprocess(image, patch_grid=(1, 1), image_size=448, use_thumbnail=False):
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

def load_image_from_image(image_file, input_size=448, patch_grid=(1, 1), use_thumbnail=True):
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

def reconstruct_to_original_size(
    patches,
    patch_grid=(1, 1),
    patch_size=448,
    orig_size=None  # (H, W)
):
    """
    Reconstruct image from normalized patches and optionally resize to original size.
    """
    # Reconstruct resized image from patches
    num_x, num_y = patch_grid
    assert patches.shape[0] == num_x * num_y

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    patches = unnormalize(patches, mean, std)

    rows = []
    for y in range(num_y):
        row = torch.cat(
            [patches[y * num_x + x] for x in range(num_x)], dim=2
        )
        rows.append(row)
    full_image = torch.cat(rows, dim=1)

    # Optionally resize back to original size
    if orig_size is not None:
        full_image = full_image.unsqueeze(0)  # (1, 3, H, W)
        full_image = F.interpolate(
            full_image, size=orig_size, mode="bicubic", align_corners=False
        )
        full_image = full_image.squeeze(0)  # (3, H_orig, W_orig)

    return full_image

def unnormalize(tensor, mean, std):
    # Reverse normalization
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def instantiate_vlm_ensemble(
    model_strs: List[str],
    model_generation_kwargs: Dict[str, Dict[str, Any]],
    accelerator: Accelerator,
) -> VLMEnsemble:
    # TODO: This function is probably overengineered and should be deleted.
    vlm_ensemble = VLMEnsemble(
        model_strs=model_strs,
        model_generation_kwargs=model_generation_kwargs,
        accelerator=accelerator,
    )
    vlm_ensemble = accelerator.prepare([vlm_ensemble])[0]
    return vlm_ensemble


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def load_jailbreak_dicts_list(
    wandb_attack_run_id: str = None,
    wandb_sweep_id: str = None,
    data_dir_path: str = "eval_data",
    refresh: bool = False,
) -> List[Dict[str, Any]]:
    os.makedirs(data_dir_path, exist_ok=True)
    runs_jailbreak_dict_list_path = os.path.join(
        data_dir_path,
        f"runs_jailbreak_dict_list_sweep={wandb_attack_run_id}.joblib",
    )
    if refresh or not os.path.exists(runs_jailbreak_dict_list_path):
        print("Downloading jailbreak images...")

        api = wandb.Api()
        if wandb_sweep_id is None and wandb_attack_run_id is not None:
            run = api.run(f"universal-vlm-jailbreak/{wandb_attack_run_id}")
            runs = [run]
        elif wandb_sweep_id is not None and wandb_attack_run_id is None:
            sweep = api.sweep(f"universal-vlm-jailbreak/{wandb_attack_run_id}")
            runs = list(sweep.runs)
        else:
            raise ValueError(
                "Invalid wandb_sweep_id and wandb_attack_run_id: "
                f"{wandb_sweep_id}, {wandb_attack_run_id}"
            )
        runs_jailbreak_dict_list = []
        for run in runs:
            for file in run.files():
                file_name = str(file.name)
                if not file_name.endswith(".png"):
                    continue
                file_dir_path = os.path.join(data_dir_path, run.id)
                os.makedirs(file_dir_path, exist_ok=True)
                file.download(root=file_dir_path, replace=True)
                # Example:
                #   'eval_data/sweep=7v3u4uq5/dz2maypg/media/images/jailbreak_image_step=500_0_6bff027c89aa794cfb3b.png'
                # becomes
                #   500
                optimizer_step_counter = int(file_name.split("_")[2][5:])
                file_path = os.path.join(file_dir_path, file_name)
                runs_jailbreak_dict_list.append(
                    {
                        "file_path": file_path,
                        "wandb_attack_run_id": run.id,
                        "optimizer_step_counter": optimizer_step_counter,
                        "models_to_attack": run.config["models_to_attack"],
                    }
                )

                print(
                    "Downloaded jailbreak image for run: ",
                    run.id,
                    " at optimizer step: ",
                    optimizer_step_counter,
                )

        # Sort runs_jailbreak_dict_list based on wandb_attack_run_id and then n_gradient_steps.
        runs_jailbreak_dict_list = sorted(
            runs_jailbreak_dict_list,
            key=lambda x: (x["wandb_attack_run_id"], x["optimizer_step_counter"]),
        )

        joblib.dump(
            value=runs_jailbreak_dict_list,
            filename=runs_jailbreak_dict_list_path,
        )

        print("Saved runs_jailbreak_dict_list to: ", runs_jailbreak_dict_list_path)

    else:
        runs_jailbreak_dict_list = joblib.load(runs_jailbreak_dict_list_path)

        print("Loaded runs_jailbreak_dict_list from: ", runs_jailbreak_dict_list_path)

    return runs_jailbreak_dict_list


def retrieve_wandb_username() -> str:
    # system_username = getpass.getuser()
    # if system_username == "rschaef":
    #     wandb_username = "rylan"
    # else:
    #     raise ValueError(f"Unknown W&B username: {system_username}")
    import wandb

    api = wandb.Api(timeout=30)
    wandb_username = api.viewer.username
    return wandb_username


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = False
        cudnn.deterministic = True
    except ImportError:
        pass

def cog_reverse_image(
    tensor: torch.Tensor,
    image_size: Optional[int] = None
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


def preprocess_model_image(model_name: str, image: PILImage, image_size=None):
    if "cog" in model_name:
        width, height = image.size
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
        cog_image: torch.Tensor = transform(image)
        return cog_image
    elif "MiniCPM" in model_name:
        width, height = image.size
        max_dim = max(width, height)
        pad_width = (max_dim - width) // 2
        pad_height = (max_dim - height) // 2
        
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
        cpm_image: torch.Tensor = transform(image)

        processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
        do_pad = True
        max_slice_nums = None
        return_tensors = TensorType.PYTORCH
        processed_image =processor.image_processor([[cpm_image]], do_pad=do_pad, max_slice_nums=max_slice_nums, return_tensors=return_tensors)
        pixel_values = processed_image["pixel_values"][0][0]
        return pixel_values
    elif "Intern" in model_name:
        width, height = image.size
        max_dim = max(width, height)
        pad_width = (max_dim - width) // 2
        pad_height = (max_dim - height) // 2
        transform_pil_image = torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.Pad(
                    (pad_width, pad_height, pad_width, pad_height), fill=0
                ),
                torchvision.transforms.v2.Resize(
                    (image_size, image_size)
                ),
            ]
        )
        intern_image = transform_pil_image(image)
        intern_image = load_image_from_image(intern_image, image_size, (1,1), True).unsqueeze(0)
        return intern_image

    else:
        other_image = (
            torchvision.transforms.v2.functional.pil_to_tensor(
                image
            ).unsqueeze(0)
            / 255.0
        )
        return other_image
    
def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform