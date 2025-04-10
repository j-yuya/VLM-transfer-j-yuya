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

from src.data import VLMEnsembleTextDataset, VLMEnsembleTextDataModule
from src.models.ensemble import VLMEnsemble
from src.image_handling import get_list_image


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
