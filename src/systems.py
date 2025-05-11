from enum import Enum, auto
import numpy as np
import lightning
import torch
import torch.optim
import torchvision.transforms
from typing import Any, Dict, List, Optional, Tuple
import wandb

from src.models.ensemble import VLMEnsemble
from src.models.evaluators import HarmBenchEvaluator, LlamaGuard2Evaluator
from src.utils import create_initial_image, create_intern_image, reconstruct_to_original_size, create_cog_image, cog_reverse_image, create_minicpm_image, reconstruct_cpm_image, preprocess_model_image


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
        self.vlm_ensemble = VLMEnsemble(
            model_strs=wandb_config["models_to_attack"],
            model_generation_kwargs=wandb_config["model_generation_kwargs"],
            regularization_args=regularization_kwargs,
            precision=wandb_config["lightning_kwargs"]["precision"],
            image_size=wandb_config["image_kwargs"]["image_size"]
        )
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
                        torch.einsum("bd,d->b", h, intern_model.r).pow(2)   # (B,)
                        for h in intern_model._hidden_layers
                    ]
                    reg_loss = torch.mean(torch.stack(proj_sq))    # scalar
                    total_loss = (1 - intern_model.beta) * loss_val + intern_model.beta * reg_loss
                    losses_per_model[loss_str] = total_loss
                elif loss_str=="MiniCPM-V-2_6":
                    cpm = self.vlm_ensemble.vlms_dict[loss_str]
                    # self._hidden_layers has one tensor per layer
                    proj_sq = [
                        torch.einsum("bd,d->b", h, cpm.r).pow(2)   # (B,)
                        for h in cpm._hidden_layers
                    ]
                    reg_loss = torch.mean(torch.stack(proj_sq))     # scalar
                    total_loss = (1 - cpm.beta) * loss_val + cpm.beta * reg_loss
                    losses_per_model[loss_str] = total_loss
                self.log(
                    f"loss/{loss_str}",
                    loss_val.detach().item(),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
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
            print(self.tensor_image.shape)

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
        super().optimizer_step(*args, **kwargs)
        self.optimizer_step_counter += 1
        with torch.no_grad():
            self.tensor_image.data = self.tensor_image.data.clamp(min=0.0, max=1.0)


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
