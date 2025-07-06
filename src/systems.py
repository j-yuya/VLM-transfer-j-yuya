from enum import Enum, auto
import numpy as np
import lightning
import torch
import torch.optim
import torchvision.transforms
from typing import Any, Dict, List, Optional, Tuple
import wandb
import random
from src.models.ensemble import VLMEnsemble
from src.models.evaluators import HarmBenchEvaluator, LlamaGuard2Evaluator
from src.utils import create_initial_image, create_intern_image, reconstruct_to_original_size, create_cog_image, cog_reverse_image, create_minicpm_image, reconstruct_cpm_image, preprocess_model_image, load_image_from_image, preprocess_cog_image,  preprocess_intern_image, preprocess_minicpm_image
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
import torchvision.transforms as T
to_tensor = T.ToTensor()  

class AttackType(Enum):
    UNCONSTRAINED = "unconstrained"
    PGD = "pgd"
    APGD = "apgd"


class VLMEnsembleAttackingSystem(lightning.LightningModule):
    def __init__(
        self,
        wandb_config: Dict[str, Any],
    ):
        super().__init__()
        tgt_sizes = None
        self.wandb_config = wandb_config
        regularization_kwargs = None
        self.use_steering_reg = False
        if "regularization_kwargs" in wandb_config.keys():
            regularization_kwargs = wandb_config["regularization_kwargs"]
            self.use_steering_reg = wandb_config["regularization_kwargs"]["use_steering_reg"]
        self.patch_cfg   = wandb_config.get("patch_attack_kwargs", {})
        self.use_patch   = bool(self.patch_cfg.get("enable", False))
        if any("Intern"  in m for m in wandb_config["models_to_attack"]):
            self._model_family = "intern"
        elif any("cogvlm2" in m for m in wandb_config["models_to_attack"]):
            self._model_family = "cog"
        elif any("MiniCPM"  in m for m in wandb_config["models_to_attack"]):
            self._model_family = "cpm"
        else:
            self._model_family = "generic"

        self.vlm_ensemble = VLMEnsemble(
            model_strs=wandb_config["models_to_attack"],
            model_generation_kwargs=wandb_config["model_generation_kwargs"],
            regularization_args=regularization_kwargs,
            precision=wandb_config["lightning_kwargs"]["precision"],
            image_size=wandb_config["image_kwargs"]["image_size"]
        )
        self.orig_H = wandb_config["image_kwargs"]["image_size"]
        self.orig_W = wandb_config["image_kwargs"]["image_size"]
        if any("Intern" in model for model in wandb_config["models_to_attack"]):
            tensor_image: torch.Tensor = create_intern_image(
            image_kwargs=wandb_config["image_kwargs"],
            seed=wandb_config["seed"],
            )
        elif any("cogvlm2" in model for model in wandb_config["models_to_attack"]):
            tensor_image: torch.Tensor = create_cog_image(
            image_kwargs=wandb_config["image_kwargs"],
            seed=wandb_config["seed"],
            )
        elif any("MiniCPM" in model for model in wandb_config["models_to_attack"]):
            tensor_image, tgt_sizes = create_minicpm_image(
            image_kwargs=wandb_config["image_kwargs"],
            seed=wandb_config["seed"],
            )
        else:
            # Load initial image plus prompt and target data.
            tensor_image: torch.Tensor = create_initial_image(
                image_kwargs=wandb_config["image_kwargs"],
                seed=wandb_config["seed"],
            )
        # print(f"tensor_image.shape: {tensor_image.shape}")
        # print(f"tensor_image: {tensor_image}")
        if tgt_sizes != None:
            self.tgt_sizes = tgt_sizes
        self.tensor_image = torch.nn.Parameter(tensor_image, requires_grad=True)
        self.convert_tensor_to_pil_image = torchvision.transforms.ToPILImage()
        self.optimizer_step_counter = 0

        if self.use_patch:
            # **Never** touch self.tensor_image for geometry – it has already
            # been pre-processed / resized.
            H, W = self.orig_H, self.orig_W

            psize = int(self.patch_cfg.get("patch_size", 32))

            if self.patch_cfg.get("center_xy") not in (None, (None, None)):
                # user-supplied absolute px coords
                cx, cy = map(int, self.patch_cfg["center_xy"])
            else:
                # helper locations (extend as you add more presets)
                loc = self.patch_cfg.get("location", "")
                if loc == "mid_top_left_quarter":
                    cx, cy = W // 4, H // 4
                else:
                    raise ValueError(f"patch_attack: unknown location '{loc}'")

            half = psize // 2
            x1, x2 = max(0, cx - half), min(W, cx + half)
            y1, y2 = max(0, cy - half), min(H, cy + half)

            m = torch.zeros((1, 1, H, W),
                            dtype=self.tensor_image.dtype,
                            device=self.tensor_image.device)
            m[:, :, y1:y2, x1:x2] = 1.0
            self.register_buffer("patch_mask_rgb", m)


    def _tensor2rgb(self, tens: torch.Tensor) -> torch.Tensor:
        fam = self._model_family

        if fam == "intern":
            return reconstruct_to_original_size(tens.squeeze(0))            # (3,H,W)

        elif fam == "cog":
            pil = cog_reverse_image(
                tens, self.wandb_config["image_kwargs"]["image_size"]
            )                                                               # PIL.Image
            return to_tensor(pil)                                           # (3,H,W)

        elif fam == "cpm":
            return reconstruct_cpm_image(tens, self.tgt_sizes)              # (3,H,W)

        else:   # generic / fallback
            return tens[0] if tens.dim() == 4 else tens                     # (3,H,W)

    def _rgb2tensor(self, rgb: torch.Tensor) -> torch.Tensor:
        model_str    = list(self.wandb_config["models_to_attack"])[0]
        image_kwargs = self.wandb_config["image_kwargs"]

        # — MiniCPM-V-2_6 ———————————————————————————————————————
        if "MiniCPM" in model_str:
            pixel_vals, tgt_sizes = preprocess_minicpm_image(rgb, image_kwargs)
            self.tgt_sizes        = tgt_sizes
            return pixel_vals                     # (3,14,14336)

        # — CogVLM-2 (Llama-3) ——————————————————————————————
        elif "cogvlm2" in model_str:
            return preprocess_cog_image(rgb, image_kwargs)  # (3,1344,1344)

        # — InternVL-2-8B ———————————————————————————————
        elif "Intern" in model_str:
            return preprocess_intern_image(rgb, image_kwargs)  # (1,…,3,P,P)

        # — Generic / fallback ————————————————————————————
        else:
            return rgb.clone() 
        
    def configure_optimizers(self) -> Dict:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        # TODO: Maybe add SWA
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.StochasticWeightAveraging.html#pytorch_lightning.callbacks.StochasticWeightAveraging
        optimization_kwargs = self.wandb_config["optimization"]
        if optimization_kwargs["optimizer"] == "adadelta":
            optimizer = torch.optim.Adadelta(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
            )
        elif optimization_kwargs["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                eps=optimization_kwargs[
                    "eps"
                ],  # https://stackoverflow.com/a/42420014/4570472
            )
        elif optimization_kwargs["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                eps=optimization_kwargs[
                    "eps"
                ],  # https://stackoverflow.com/a/42420014/4570472
            )
        elif optimization_kwargs["optimizer"] == "rmsprop":
            optimizer = torch.optim.RMSprop(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                momentum=optimization_kwargs["momentum"],
                eps=1e-4,
            )
        elif optimization_kwargs["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                momentum=optimization_kwargs["momentum"],
            )
        else:
            # TODO: add adafactor https://pytorch-optimizer.readthedocs.io/en/latest/index.html
            raise NotImplementedError(f"{self.wandb_config['optimizer']}")

        optimizer_and_maybe_others_dict = {
            "optimizer": optimizer,
        }

        # if self.wandb_config["learning_rate_scheduler"] is None:
        #     pass
        # elif (
        #     self.wandb_config["learning_rate_scheduler"]
        #     == "cosine_annealing_warm_restarts"
        # ):
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #         optimizer=optimizer,
        #         T_0=2,
        #     )
        #     optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
        #
        # elif (
        #     self.wandb_config["learning_rate_scheduler"]
        #     == "linear_warmup_cosine_annealing"
        # ):
        #     from flash.core.optimizers import LinearWarmupCosineAnnealingLR
        #
        #     scheduler = LinearWarmupCosineAnnealingLR(
        #         optimizer=optimizer,
        #         warmup_epochs=1,
        #         max_epochs=self.wandb_config["n_epochs"],
        #     )
        #
        #     optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
        #
        # elif self.wandb_config["learning_rate_scheduler"] == "reduce_lr_on_plateau":
        #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         factor=0.95,
        #         optimizer=optimizer,
        #         patience=3,
        #     )
        #     optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
        #     optimizer_and_maybe_others_dict["monitor"] = "train/loss=total_loss"
        # else:
        #     raise NotImplementedError(f"{self.wandb_config['learning_rate_scheduler']}")

        return optimizer_and_maybe_others_dict

    def training_step(
        self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training_step
        if self.use_steering_reg:
            if "MiniCPM-V-2_6" in self.vlm_ensemble.vlms_dict.keys():
                self.vlm_ensemble.vlms_dict["MiniCPM-V-2_6"]._hidden_layers.clear()
            elif "InternVL2-8B" in self.vlm_ensemble.vlms_dict.keys():
                self.vlm_ensemble.vlms_dict["InternVL2-8B"]._hidden_layers.clear()
        losses_per_model: Dict[str, torch.Tensor] = self.vlm_ensemble.compute_loss(
            image=self.tensor_image,
            text_data_by_model=batch,
        )
        for loss_str, loss_val in losses_per_model.items():
            if self.use_steering_reg:
                if loss_str=="InternVL2-8B":
                    intern_model = self.vlm_ensemble.vlms_dict[loss_str]
                    proj_sq = [
                        torch.einsum("bd,d->b", h, intern_model.r).pow(2)    # (B,)
                        for h in intern_model._hidden_layers
                    ]
                    reg_loss = torch.stack([t.mean() for t in proj_sq]).mean()  
                    regularization_factor = intern_model.beta * reg_loss
                    ce_factor = (1 - intern_model.beta) * loss_val
                    total_loss =  ce_factor + regularization_factor 
                    losses_per_model[loss_str] = total_loss
                elif loss_str=="MiniCPM-V-2_6":
                    cpm = self.vlm_ensemble.vlms_dict[loss_str]
                    # self._hidden_layers has one tensor per layer
                    proj_sq = [
                        torch.einsum("bd,d->b", h, cpm.r).pow(2)    # (B,)
                        for h in cpm._hidden_layers
                    ]
                    reg_loss = torch.stack([t.mean() for t in proj_sq]).mean()  
                    regularization_factor = cpm.beta * reg_loss
                    ce_factor = (1 - cpm.beta) * loss_val
                    total_loss =  ce_factor + regularization_factor 
                    losses_per_model[loss_str] = total_loss
                self.log(
                    f"loss/{loss_str}",
                    total_loss.detach().item(),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
                self.log(
                    f"loss_reg/{loss_str}",
                    regularization_factor.detach().item(),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
                self.log(
                    f"loss_reg_unweighted/{loss_str}",
                    reg_loss.detach().item(),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
                self.log(
                    f"loss_ce/{loss_str}",
                    ce_factor.detach().item(),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
                losses_per_model["avg"] = torch.mean(
                    torch.stack(list(losses_per_model.values()))
                )
            else:
                self.log(
                    f"loss/{loss_str}",
                    loss_val.detach().item(),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )

        self.log(
            "optimizer_step_counter",
            self.optimizer_step_counter,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # if batch_idx == 0:
        #     print(torch.cuda.memory_summary())
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass

        return losses_per_model["avg"]

    def optimizer_step(self, *args, **kwargs):
        if (
            self.optimizer_step_counter
            % self.wandb_config["lightning_kwargs"]["log_image_every_n_steps"]
        ) == 0:
            # TODO: Add handling for additional image dim (4)

            log_image = self.tensor_image.detach().cpu()
            if any("Intern" in model for model in self.wandb_config["models_to_attack"]):
                wandb.log(
                    {
                        f"jailbreak_image_step={self.optimizer_step_counter}": wandb.Image(
                            # https://docs.wandb.ai/ref/python/data-types/image
                            # 0 removes the size-1 batch dimension.
                            # The transformation doesn't accept bfloat16.
                            data_or_path=self.convert_tensor_to_pil_image(
                                reconstruct_to_original_size(log_image.squeeze(0)).to(torch.float32)
                            ),
                            # caption="Adversarial Image",
                        ),
                    },
                )
            elif any("cogvlm2" in model for model in self.wandb_config["models_to_attack"]):
                wandb.log(
                    {
                        f"jailbreak_image_step={self.optimizer_step_counter}": wandb.Image(
                            # https://docs.wandb.ai/ref/python/data-types/image
                            # 0 removes the size-1 batch dimension.
                            # The transformation doesn't accept bfloat16.
                            data_or_path=cog_reverse_image(log_image, self.wandb_config["image_kwargs"]["image_size"])
                            # caption="Adversarial Image",
                        ),
                    },
                )
            elif any("MiniCPM" in model for model in self.wandb_config["models_to_attack"]):
                rec_image = reconstruct_cpm_image(log_image, self.tgt_sizes)
                print(rec_image.shape)
                wandb.log(
                    {
                        f"jailbreak_image_step={self.optimizer_step_counter}": wandb.Image(
                            # https://docs.wandb.ai/ref/python/data-types/image
                            # 0 removes the size-1 batch dimension.
                            # The transformation doesn't accept bfloat16.
                            data_or_path=self.convert_tensor_to_pil_image(
                                reconstruct_cpm_image(log_image, self.tgt_sizes).to(torch.float32)
                            ),
                            # caption="Adversarial Image",
                        ),
                    },
                )
            else:
                wandb.log(
                    {
                        f"jailbreak_image_step={self.optimizer_step_counter}": wandb.Image(
                            # https://docs.wandb.ai/ref/python/data-types/image
                            # 0 removes the size-1 batch dimension.
                            # The transformation doesn't accept bfloat16.
                            data_or_path=self.convert_tensor_to_pil_image(
                                log_image[0].to(torch.float32)
                            ),
                            # caption="Adversarial Image",
                        ),
                    },
                )
        if self.use_patch:
            pre_step_tensor = self.tensor_image.detach().clone()
        super().optimizer_step(*args, **kwargs)
        self.optimizer_step_counter += 1
        if self.use_patch:
            with torch.no_grad():
                rgb_before = self._tensor2rgb(pre_step_tensor)
                rgb_after  = self._tensor2rgb(self.tensor_image)

                mask     = self.patch_mask_rgb          # (1,1,H,W)
                rgb_new  = rgb_before * (1 - mask.squeeze(1)) + rgb_after * mask.squeeze(1)

                # ⬇️  ONE universal clamp, no model-specific logic needed
                rgb_new.clamp_(0.0, 1.0)

                # back to model space
                # print(f"rgb_before shape : {rgb_before.shape}")   # (3,H,W)
                # print(f"rgb_after  shape : {rgb_after.shape}")    # (3,H,W)
                # print(f"mask shape       : {mask.shape}")         # (1,1,H,W)

                # # Mask sanity: how many pixels belong to the patch?
                # num_patch_px = mask.sum().item()
                # H, W = rgb_before.shape[-2:]
                # print(f"patch size       : {num_patch_px} / {H*W} px "
                #     f"({100*num_patch_px/(H*W):.2f} %)")

                # # Channel-wise statistics *inside* and *outside* the patch
                # m = mask.squeeze(1)                     # (1,H,W)
                # m3 = m.expand_as(rgb_before)            # broadcast to (3,H,W)

                # def stats(name, tensor):
                #     print(f"{name:11}  min={tensor.min():.4f}  "
                #         f"mean={tensor.mean():.4f}  max={tensor.max():.4f}")

                # stats("before in",  rgb_before[m3.bool()])
                # stats("after  in",  rgb_after[m3.bool()])
                # stats("before out", rgb_before[~m3.bool()])
                # stats("after  out", rgb_after[~m3.bool()])

                # # How much did the optimiser change the patch this step?
                # delta = (rgb_after - rgb_before) * m3
                # stats("Δ patch", delta)

                # # (Optional) check for bad values *before* the RGB clamp
                # bad = (rgb_after < 0) | (rgb_after > 1)
                # if bad.any():
                #     bad_pct = 100 * bad.float().mean().item()
                #     print(f"⚠️  {bad_pct:.2f}% of patch pixels are outside [0,1]"
                #         " BEFORE rgb clamp")
                self.tensor_image.data.copy_(self._rgb2tensor(rgb_new))
                # import pdb
                # pdb.set_trace()
        else:
            with torch.no_grad():
                if "InternVL2-8B" in self.vlm_ensemble.vlms_dict.keys():
                    IMAGENET_MEAN = (0.485, 0.456, 0.406)
                    IMAGENET_STD  = (0.229, 0.224, 0.225)

                    
                    bounds = [(0.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
                    _min   = torch.tensor(bounds, device=self.tensor_image.device, dtype=self.tensor_image.dtype)[None, :, None, None]
                    _max   = torch.tensor([(1.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
                                        device=self.tensor_image.device, dtype=self.tensor_image.dtype)[None, :, None, None]
                elif "MiniCPM-V-2_6" in self.vlm_ensemble.vlms_dict.keys():
                    CPM_MEAN = (0.5, 0.5, 0.5)
                    CPM_STD  = (0.5, 0.5, 0.5)

                    device = self.tensor_image.device
                    dtype  = self.tensor_image.dtype

                    mean = torch.tensor(CPM_MEAN, device=device, dtype=dtype)[:, None, None]  # (3,1,1)
                    std  = torch.tensor(CPM_STD,  device=device, dtype=dtype)[:, None, None]  # (3,1,1)

                    _min = (0.0 - mean) / std        # (3,1,1)  → broadcast to (3,14,14336)
                    _max = (1.0 - mean) / std
                elif "cogvlm2-llama3-chat-19B" in self.vlm_ensemble.vlms_dict:
                    COG_MEAN = (0.48145466, 0.4578275, 0.40821073)
                    COG_STD  = (0.26862954, 0.26130258, 0.27577711)

                    mean = torch.tensor(COG_MEAN, device=self.tensor_image.device, dtype=self.tensor_image.dtype)[:, None, None]
                    std  = torch.tensor(COG_STD,  device=self.tensor_image.device, dtype=self.tensor_image.dtype)[:, None, None]

                    _min = (0.0 - mean) / std
                    _max = (1.0 - mean) / std
                else:
                    _min = 0.0
                    _max = 1.0

                self.tensor_image.data = self.tensor_image.data.clamp(min=_min, max=_max)

            # dev, dtype = self.tensor_image.device, self.tensor_image.dtype
            # img_denorm = unnormalize(self.tensor_image.squeeze(0)).clamp_(0, 1) 
            # # print(img_denorm.shape)
            # # img_orig = reconstruct_to_original_size_2(img_denorm.cpu(), is_normalised=False)
            # # img_proj = project_to_low_res(img_orig.unsqueeze(0), 448, 224)
            # patches = make_single_patch(img_denorm).to(dev, dtype=dtype)      
            # self.tensor_image.data.copy_(patches)  


def make_single_patch(img: torch.Tensor) -> torch.Tensor:
    """
    img: (3,448,448) in [0,1]  – already projected to low res
    returns: (1,3,448,448) normalised with ImageNet mean/std  (on same device)
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    mean = torch.tensor(IMAGENET_MEAN, device=img.device,
                        dtype=img.dtype).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=img.device,
                        dtype=img.dtype).view(3,1,1)
    return ((img - mean) / std).unsqueeze(0)          

def project_to_low_res(img: torch.Tensor, HIGH, LOW) -> torch.Tensor:
    """
    img: (B, 3, 448, 448) in [0,1]  (or whatever range you use)
    returns: same shape, but with only LOW×LOW degrees of freedom
    """
    print(img.shape)
    # ↓ 1) shrink to LOW×LOW (bilinear keeps it differentiable)
    img_small = F.interpolate(img, size=(LOW, LOW),
                              mode='bilinear', align_corners=False, antialias=True)

    # ↑ 2) blow it back up with *nearest* so every 8×8 block is constant
    img_blocky = F.interpolate(img_small, size=(HIGH, HIGH), mode='nearest')
    return img_blocky

def unnormalize(img: torch.Tensor):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    mean = torch.tensor(IMAGENET_MEAN, device=img.device,
                        dtype=img.dtype).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=img.device,
                        dtype=img.dtype).view(3,1,1)
    return img * std + mean

def reconstruct_to_original_size_2(
    patches: torch.Tensor,              # (P,3,H,W)  *or*  (3,H,W)
    patch_grid=(1, 1),
    patch_size=448,
    orig_size=None,                     # (H_orig, W_orig)  or None
    is_normalised=True,                 # set False if already un-norm’d
):
    """
    Stitch `num_x × num_y` patches back to a single image and optionally
    resize to `orig_size`.

    • `patch_grid` must correspond to how you split the image earlier.
    • Accepts both batched and single-patch tensors.
    """
    # 0) ensure shape is (P,3,H,W)
    if patches.dim() == 3:                           # (3,H,W) → add batch dim
        patches = patches.unsqueeze(0)
    elif patches.dim() != 4 or patches.shape[1] != 3:
        raise ValueError("Expected (P,3,H,W) or (3,H,W) tensor.")

    num_x, num_y = patch_grid
    assert patches.shape[0] == num_x * num_y, (
        f"Grid {patch_grid} expects {num_x*num_y} patches, "
        f"but got {patches.shape[0]}.")

    # 1) (optional) un-normalise
    if is_normalised:
        patches = unnormalize(patches)

    # 2) stitch row by row
    rows = []
    for y in range(num_y):
        row = torch.cat(
            [patches[y * num_x + x] for x in range(num_x)], dim=2  # concat W
        )                      # → (3, H, num_x*W)
        rows.append(row)
    full_image = torch.cat(rows, dim=1)               # concat H  → (3, H*ny, W*nx)

    # 3) optional resize back to original resolution
    if orig_size is not None:
        full_image = F.interpolate(
            full_image.unsqueeze(0),                  # add batch
            size=orig_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

    return full_image   # (3, H_out, W_out)


# class VLMEnsembleAttackingSystem2(lightning.LightningModule):
#     def __init__(
#         self,
#         wandb_config: Dict[str, Any],
#     ):
#         super().__init__()
#         self.first_step = True
#         tgt_sizes = None
#         self.wandb_config = wandb_config
#         regularization_kwargs = None
#         self.use_steering_reg = False
#         if "regularization_kwargs" in wandb_config.keys():
#             regularization_kwargs = wandb_config["regularization_kwargs"]
#             self.use_steering_reg = wandb_config["regularization_kwargs"]["use_steering_reg"]
#         self.vlm_ensemble = VLMEnsemble(
#             model_strs=wandb_config["models_to_attack"],
#             model_generation_kwargs=wandb_config["model_generation_kwargs"],
#             regularization_args=regularization_kwargs,
#             precision=wandb_config["lightning_kwargs"]["precision"],
#             image_size=wandb_config["image_kwargs"]["image_size"]
#         )
#         if any("Intern" in model for model in wandb_config["models_to_attack"]):
#             tensor_image: torch.Tensor = create_intern_image(
#             image_kwargs=wandb_config["image_kwargs"],
#             seed=wandb_config["seed"],
#             )
#         elif any("cogvlm2" in model for model in wandb_config["models_to_attack"]):
#             tensor_image: torch.Tensor = create_cog_image(
#             image_kwargs=wandb_config["image_kwargs"],
#             seed=wandb_config["seed"],
#             )
#         elif any("MiniCPM" in model for model in wandb_config["models_to_attack"]):
#             tensor_image, tgt_sizes = create_minicpm_image(
#             image_kwargs=wandb_config["image_kwargs"],
#             seed=wandb_config["seed"],
#             )
#         else:
#             # Load initial image plus prompt and target data.
#             tensor_image: torch.Tensor = create_initial_image(
#                 image_kwargs=wandb_config["image_kwargs"],
#                 seed=wandb_config["seed"],
#             )
#         # print(f"tensor_image.shape: {tensor_image.shape}")
#         # print(f"tensor_image: {tensor_image}")
#         if tgt_sizes != None:
#             self.tgt_sizes = tgt_sizes
#         self.tensor_image = torch.nn.Parameter(tensor_image, requires_grad=True)
#         self.convert_tensor_to_pil_image = torchvision.transforms.ToPILImage()
#         self.optimizer_step_counter = 0

#     def configure_optimizers(self) -> Dict:
#         return None

#     def predict_step(
#         self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int
#     ) -> torch.Tensor:
#         # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training_step
#         if self.use_steering_reg:
#             if "MiniCPM-V-2_6" in self.vlm_ensemble.vlms_dict.keys():
#                 self.vlm_ensemble.vlms_dict["MiniCPM-V-2_6"]._hidden_layers.clear()
#             elif "InternVL2-8B" in self.vlm_ensemble.vlms_dict.keys():
#                 self.vlm_ensemble.vlms_dict["InternVL2-8B"]._hidden_layers.clear()
#         if self.first_step:
#             losses_per_model: Dict[str, torch.Tensor] = self.vlm_ensemble.compute_loss(
#                 image=self.tensor_image,
#                 text_data_by_model=batch,
#             )
#             self.first_step = False
#             for loss_str, loss_val in losses_per_model.items():
#                 if self.use_steering_reg:
#                     if loss_str=="InternVL2-8B":
#                         intern_model = self.vlm_ensemble.vlms_dict[loss_str]
#                         proj_sq = [
#                             torch.einsum("bd,d->b", h, intern_model.r).pow(2)   # (B,)
#                             for h in intern_model._hidden_layers
#                         ]
#                         reg_loss = torch.mean(torch.stack(proj_sq))    # scalar
#                         total_loss = (1 - intern_model.beta) * loss_val + intern_model.beta * reg_loss
#                         losses_per_model[loss_str] = total_loss
#                     elif loss_str=="MiniCPM-V-2_6":
#                         cpm = self.vlm_ensemble.vlms_dict[loss_str]
#                         # self._hidden_layers has one tensor per layer
#                         proj_sq = [
#                             torch.einsum("bd,d->b", h, cpm.r).pow(2)   # (B,)
#                             for h in cpm._hidden_layers
#                         ]
#                         reg_loss = torch.mean(torch.stack(proj_sq))     # scalar
#                         total_loss = (1 - cpm.beta) * loss_val + cpm.beta * reg_loss
#                         losses_per_model[loss_str] = total_loss
#                     self.log(
#                         f"loss/{loss_str}",
#                         loss_val.detach().item(),
#                         on_step=True,
#                         on_epoch=False,
#                         sync_dist=True,
#                     )
#                 else:
#                     self.log(
#                         f"loss/{loss_str}",
#                         loss_val.detach().item(),
#                         on_step=True,
#                         on_epoch=False,
#                         sync_dist=True,
#                     )
#         for _ in range(self.n_iters):
#             improved_any = False

#             for _ in range(self.tries):
#                 rnd_words = random.choices(self.word_list, k=len(imgs))

#                 # build batch of candidate images + remember their meta
#                 cand_imgs, meta = [], []
#                 for i in range(len(imgs)):
#                     img_, info = self.overlay_random_word(best_imgs[i].cpu(), rnd_words[i])
#                     cand_imgs.append(img_)
#                     meta.append(info)
#                 cand_imgs = torch.stack(cand_imgs).to(device)

#                 # evaluate
#                 with torch.no_grad():
#                     cand_loss = self.loss_fn(self.vlm(cand_imgs, prompts), targets)

#                 # decide improvement per-example
#                 better = cand_loss > best_loss if self.maximise else cand_loss < best_loss
#                 if better.any():
#                     improved_any = True
#                     best_imgs[better] = cand_imgs[better]
#                     best_loss[better] = cand_loss[better]
#                     # store meta only on improved ones
#                     for idx, flag in enumerate(better.tolist()):
#                         if flag:
#                             histories[idx].append(meta[idx])
#                     break   # restart tries because state changed

#             if not improved_any:          # no improvement in an entire outer-loop: stop early
#                 break

#         self.log(
#             "optimizer_step_counter",
#             self.optimizer_step_counter,
#             on_step=True,
#             on_epoch=False,
#             sync_dist=True,
#         )

#         # if batch_idx == 0:
#         #     print(torch.cuda.memory_summary())
#         # for obj in gc.get_objects():
#         #     try:
#         #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#         #             print(type(obj), obj.size())
#         #     except:
#         #         pass

#         return losses_per_model["avg"]

    
#     def overlay_random_word(
#         self,
#         img: torch.Tensor,
#         word: str,
#         font: Optional[ImageFont.ImageFont] = None,
#         angle_range: Tuple[int, int] = (-45, 45),
#         scale_range: Tuple[int, int] = (14, 40),
#     ) -> torch.Tensor:
#         """
#         Draw `word` on a CHW float tensor in [0,1] with random position, scale, angle.
#         Returns *new* tensor; original is unchanged.
#         """
#         c, h, w = img.shape
#         pil_base = TF.to_pil_image(img).convert("RGBA")

#         # --- prepare text layer -------------------------------------------------
#         font_sz = random.randint(*scale_range)
#         if font is None:
#             font = ImageFont.load_default()
#         else:
#             font = font.font_variant(size=font_sz)

#         txt_w, txt_h = font.getsize(word)
#         txt_layer = Image.new("RGBA", (txt_w, txt_h), (0, 0, 0, 0))
#         ImageDraw.Draw(txt_layer).text((0, 0), word, fill=(255, 255, 255, 255), font=font)

#         # random rotation
#         angle = random.uniform(*angle_range)
#         txt_layer = txt_layer.rotate(angle, expand=True)

#         # random position (clip so it stays inside the canvas)
#         max_x = max(1, w - txt_layer.width)
#         max_y = max(1, h - txt_layer.height)
#         pos = (random.randint(0, max_x), random.randint(0, max_y))

#         # composite
#         pil_base.alpha_composite(txt_layer, dest=pos)
#         return TF.to_tensor(pil_base.convert("RGB")), {"word": word, "pos": pos, "angle": angle, "size": font_sz}


class VLMEnsembleEvaluatingSystem(lightning.LightningModule):
    def __init__(
        self,
        wandb_config: Dict[str, Any],
        tensor_image = None,
    ):
        super().__init__()
        self.wandb_config = wandb_config
        self.vlm_ensemble = VLMEnsemble(
            model_strs=wandb_config["model_to_eval"],
            model_generation_kwargs=wandb_config["model_generation_kwargs"],
            regularization_args=None,
            image_size=wandb_config["image_kwargs"]["image_size"]
        )
        #self.tensor_image = torch.nn.Parameter(tensor_image, requires_grad=False)
        self.tensor_image = tensor_image
        self.wandb_additional_data = {}
        self.model_strs = wandb_config["model_to_eval"]
        self.tensor_images = {}
        print(tensor_image)
        for model_str in self.model_strs:
            self.tensor_images[model_str] = torch.nn.Parameter(preprocess_model_image(model_str, tensor_image, self.wandb_config["image_kwargs"]["image_size"]), requires_grad=False)

    def update_tensor_images(self, image, image_size=None):
        for model_str in self.model_strs:
            self.tensor_images[model_str] = preprocess_model_image(model_str, image, image_size)


    def test_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int):
        if self.tensor_image is None:
            raise ValueError("Image must be provided!")

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training_step
        losses_per_model: Dict[str, torch.Tensor] = self.vlm_ensemble.compute_loss_eval(
            images=self.tensor_images,
            text_data_by_model=batch,
        )

        for loss_str, loss_val in losses_per_model.items():
            self.log(
                f"loss/{loss_str}",
                loss_val.detach().item(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        # Make sure the number of optimizer steps is simultaneously logged.
        self.log(
            "optimizer_step_counter",
            self.wandb_additional_data["optimizer_step_counter"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
