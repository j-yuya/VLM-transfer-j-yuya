# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
from src.models.qwen_utils.modeling_qwen import QWenLMHeadModel
from transformers import AutoTokenizer, AutoModel
import lightning
import torch
from typing import Any, Callable, Dict, List, Mapping, Optional
from transformers import PreTrainedTokenizer
from types import MethodType
from src.models.base import VisionLanguageModel
import torch.nn as nn

from src.models.qwen_utils.visual import VisionTransformer


# Labels with these indices will be ignored by cross entropy loss in PyTorch.
IGNORE_INDEX = -100

def get_prompt_for_internvl2(system_message: str, messages: list) -> str:
                """
                Mimics `get_prompt()` for the InternVL2 template using SeparatorStyle.MPT.

                Args:
                    system_message (str): The system prompt (usually in Chinese for InternVL2).
                    messages (list): List of (role, message) tuples like:
                                    [('<|im_start|>user\n', 'Hi there'), ('<|im_start|>assistant\n', None)]

                Returns:
                    str: Full prompt string to feed into tokenizer/model.
                """
                sep = '<|im_end|>'
                ret = f"<|im_start|>system\n{system_message}{sep}"

                for role, message in messages:
                    if message:
                        # If message is a tuple, unpack it
                        if isinstance(message, tuple):
                            message = message[0]
                        ret += f"{role}{message}{sep}"
                    else:
                        ret += role  # usually ends with assistant turn

                return ret

def format_instruction_internvl(
        instruction: str,
        output: str = None,
        include_trailing_whitespace: bool = True
    ):
        # Code not for multi-turn convs, only single-turn
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n')
        messages = []
        messages.append([roles[0], instruction])
        messages.append([roles[1], None])
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。"
        query = get_prompt_for_internvl2(system_message=system_message, messages=messages)
        formatted_instruction = query

        if not include_trailing_whitespace:
            formatted_instruction = formatted_instruction.rstrip()
        
        if output is not None:
            formatted_instruction += output

        return formatted_instruction


class InternVL2(VisionLanguageModel, lightning.LightningModule):
    def __init__(
        self,
        #TODO: More Param Variants
        model_str: str = "InternVL2-8B",
        generation_kwargs: Mapping[str, Any] | None = None,
        regularization_args=None,
        precision: str = "bf16-mixed",
        image_size: int = 448,
    ):
        super().__init__(image_size)
        self.already_logged_new_mask: bool = False  # For print debugigng
        self.already_logged_text: bool = False  # For print debugigng
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

        model_path = f"OpenGVLab/{model_str}"

        # not sure why we need to register the image processor manually
        print(f"Using InternVL2 model: {model_path}")

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # type: ignore
        # qwen doesn't have a specific pad token, but since we mask it out we can use any token
        # see https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
        self.tokenizer.padding_side = "left"
        #TODO: maybe use '<|im_end|>'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #self.pad_token_id = 55
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.precision_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        
        if regularization_args and regularization_args["use_steering_reg"]:
            self.use_steering_reg = True
            self.layer_idx=regularization_args["layer_idx"]
            self.pos_idx=regularization_args["pos_idx"]
            self._capture_hidden()
            r = torch.load(regularization_args["direction_path"])  
            self.r = (r / r.norm(dim=-1, keepdim=True)).to(dtype=torch.bfloat16, device=self.model.device)
            self.beta = regularization_args["beta"]
        #self.already_logged_new_mask: bool = False  # For print debugigng
        #self.already_logged_text: bool = False  # For print debugigng

    def create_images_transform_fn(self, model_str: str) -> Callable:
        raise NotImplementedError(
            "create_images_transform_fn is not implemented for DeepSeek models."
        )

    def compute_loss(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,  # before adding image tokens, because this model needs the image_seq_mask
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        device = self.model.device
        batch_size = input_ids.size(0)

        # seq_len  = attention_mask.sum(-1).max().item()   # longest *real* token
        # input_ids      = input_ids[:, -seq_len:]
        # attention_mask = attention_mask[:, -seq_len:]
        # labels         = labels[:, -seq_len:]
        # Since we only get a single image, we need to repeat it for the batch size.
        assert image.ndim == 5, f"Expected 4 dims, got {image.ndim}"
        # assert that we only have one image here
        assert (
            image.size(0) == 1
        ), f"Expected only 1 image that we repeat, got {image.size(0)}"
        image = image.repeat(batch_size, 1, 1, 1, 1)
        image_seq_len = image.shape[1]  # T
        image_flags = torch.tensor([1] * image_seq_len, dtype=torch.long, device=device)
        image_flags = image_flags.unsqueeze(0).repeat(batch_size, 1)

        # torch.set_printoptions(threshold=10000)
        # print("seq_len in compute_loss :", input_ids[0].numel())
        # print(f"First input_ids: {input_ids[0]}")
        # print(f"First attention_mask: {attention_mask[0]}")
        # print(f"First labels: {labels[0]}")
        with torch.autocast(device_type='cuda', dtype=self.precision_dtype):
            outputs = self.model(
                input_ids=input_ids.to(device=device),
                pixel_values=image.squeeze(0).to(device=device, dtype=self.precision_dtype),
                attention_mask=attention_mask.to(device=device),
                image_flags = image_flags.to(device=device),
                labels=labels.to(device=device),
            )
        print("LOSS")
        print(outputs.loss.item())
        return outputs.loss

    def convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        assert targets is not None, "Not support yet."

        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        NUM_IMAGE_TOKENS = 256
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id
        num_patches_list = [torch.tensor([1]) for _ in prompts]
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        prompt_texts = []

        for prompt, target in zip(prompts, targets):
            # Construct image placeholder with visual tokens
            prompt_w_image_tag = f"<image>\n{prompt}"
            query = format_instruction_internvl(prompt_w_image_tag)
            visual_token_str = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * NUM_IMAGE_TOKENS) + IMG_END_TOKEN
            query = query.replace('<image>', visual_token_str, 1)
            if target is not None:
                query = query + target + "<|im_end|>"
            #image_plus_prompt = f"{visual_token_str}\n{prompt}"
            prompt_texts.append(query)
        
        model_inputs = self.tokenizer(prompt_texts, padding=True, truncation=False, return_tensors='pt')
        results = {}
        results["input_ids"] = model_inputs["input_ids"]
        results["attention_mask"] = model_inputs["attention_mask"]
        input_ids = results["input_ids"]
        attention_mask = results["attention_mask"]
        if targets[0] is not None:
            labels = make_labels(
                input_ids=input_ids,
                pad_token_id=pad_token_id,
                targets=targets,
                tokenizer=self.tokenizer,
            )
            results["labels"] = labels

        if not self.already_logged_text:
            torch.set_printoptions(threshold=10000)
            first_text = prompt_texts[0]
            # print(f"First text: {first_text}")
            # print(f"First prmpt: {prompts[0]}")
            # print(f"First target: {targets[0]}")
            # print(f"First input_ids: {results['input_ids'][0]}")
            # print(f"First attention_mask: {results['attention_mask'][0]}")
            # print(f"First labels: {results['labels'][0]}")
            # print("seq_len in converter :", results["input_ids"][0].numel())

            # # if len(input_ids) > 1:
            #     print(f"Second input ids: {input_ids[1]}")
            #     print(f"Second attention_mask: {attention_mask[1]}")
            #     print(f"Second labels: {results['labels'][1]}")
            # non_minus_100 = [r for r in results["labels"][0] if r != IGNORE_INDEX]
            # non_minus_100_text = self.tokenizer.decode(non_minus_100)
            # print(f"Example text that we calculate loss on: {non_minus_100_text}")
            torch.set_printoptions(profile="default")
            self.already_logged_text = True
        return results

    @torch.inference_mode()
    def generate(self, image: torch.Tensor, prompts: List[str]) -> List[str]:
        # We should only have a single image.
        self.remove_hidden_hooks()
        assert image.shape[0] == 1, print(image.shape[0])
        assert image.ndim == 5, f"Expected (1, p, 3, H, W), got {image.shape}"
        # we have (1, 3, h,w) , we want (3, H, W)
        model_generations = []

        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        NUM_IMAGE_TOKENS = 256
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id
        num_patches_list = [torch.tensor([1]) for _ in prompts]
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        prompt_texts = []
        self.generation_kwargs["eos_token_id"] = eos_token_id
        for prompt in prompts:
            prompt_w_image_tag = f"<image>\n{prompt}"
            query = format_instruction_internvl(prompt_w_image_tag)
            for num_patches in num_patches_list:
                visual_token_str = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * NUM_IMAGE_TOKENS * num_patches) + IMG_END_TOKEN
                query = query.replace('<image>', visual_token_str, 1)
            model_inputs = self.tokenizer(query, padding=True, truncation=False, return_tensors='pt')
            do_sample = (
                True if self.generation_kwargs.get("temperature", 0) > 0 else False
            )

            # # run the model to get the response

            # print(f"Prompting with image: {image}")

            generation_config = self.model.generation_config
            assert (
                generation_config is not None
            ), "Expected generation config to be set."
            # # run the model to get the response
            # these stop words are the im_end, so they are the REAL eos

            outputs = self.model.generate(
                input_ids=model_inputs.input_ids.to(self.model.device),
                pixel_values=image.squeeze(0).to(torch.bfloat16).to(self.model.device),
                #attention_mask=model_inputs.attention_mask,
                # max_new_tokens=512,
                do_sample=do_sample,
                #use_cache=True,
                **self.generation_kwargs,
            )
            # print(f"Got type: {type(outputs)}")
            out: str = self.tokenizer.decode(
                outputs.squeeze(), skip_special_tokens=True
            )
            #clean_out = only_assistant_response(starting_text=prompt, response=out)

            model_generations.append(out)

        return model_generations

    def disable_model_gradients(self):       
        # Disable gradients for the full model
        self.model.requires_grad_(False)
        self.model.eval()

        # Vision model
        if hasattr(self.model, "vision_model"):
            self.model.vision_model.requires_grad_(False)
            self.model.vision_model.eval()

        # Language model
        if hasattr(self.model, "language_model"):
            self.model.language_model.requires_grad_(False)
            self.model.language_model.eval()
            if hasattr(self.model.language_model, "model"):
                self.model.language_model.model.requires_grad_(False)
                self.model.language_model.model.eval()

        # Optional: also freeze MLP if present
        if hasattr(self.model, "mlp1"):
            self.model.mlp1.requires_grad_(False)
            self.model.mlp1.eval()

    def _capture_hidden(self, layer_idx: int | None = None, track_grad = True):
        """
        If `layer_idx` is an int, capture that single layer (old behaviour).
        If None, capture *every* decoder block (like the CPM helper).
        Results:
            ─ single layer  → self._hidden   (Tensor)
            ─ all layers    → self._hidden_layers (list[Tensor])
        """
        # 1) remove previous hooks
        self.remove_hidden_hooks()

        # 2) configure storage
        if layer_idx is None:
            self._hidden_layers: list[torch.Tensor] = []
        else:
            self._hidden = None

        # 3) helper that returns a proper hook
        def make_hook(idx):
            def hook(_mod, inp):                  # ⟵ pre-block hook
                hidden = inp[0] if isinstance(inp, (tuple, list)) else inp
                vec = hidden[:, self.pos_idx, :]
                if not track_grad:
                    vec = vec.detach()

                if layer_idx is None:           # collecting many
                    self._hidden_layers.append(vec)
                elif idx == layer_idx:          # collecting one
                    self._hidden = vec
            return hook

        # 4) attach hooks
        self._iris_handles: list = []
        layers = self.model.language_model.model.layers
        if layer_idx is None:
            for i, blk in enumerate(layers):
                h = blk.register_forward_pre_hook(make_hook(i))
                self._iris_handles.append(h)
        else:
            blk = layers[layer_idx]
            h = blk.register_forward_pre_hook(make_hook(layer_idx))
            self._iris_handles.append(h)

    # <<< NEW / RENAMED >>>
    def remove_hidden_hooks(self):
        """Remove any hooks created by _capture_hidden."""
        for h in getattr(self, "_iris_handles", []):
            h.remove()
        self._iris_handles = []
        self._hidden = None
        self._hidden_layers = []

def only_assistant_response(starting_text: str, response: str) -> str:
    assert starting_text in response, f"Expected {starting_text} to be in {response}"
    # remove everything before and including the assistant token
    new_response = response.split(starting_text)[1]
    # # remove the final \n
    # new_response = new_response[:-1]
    return new_response

IGNORE_INDEX = -100  # standard for ignoring in loss

def make_labels(
    input_ids: torch.Tensor, 
    pad_token_id: int, 
    targets: list[str], 
    tokenizer
):
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX  # mask everything by default

    # Tokenize targets individually to get length
    tokenized_targets = tokenizer(targets, add_special_tokens=False).input_ids

    for i, target_ids in enumerate(tokenized_targets):
        # Locate the position where the target begins
        # Assumption: target is always at the *end* of the input, followed by <|im_end|>
        # So we place the labels on the correct slice at the end

        target_len = len(target_ids)
        # Try to find the slice [.... target tokens ..., <|im_end|>] at the end
        end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sequence = input_ids[i].tolist()

        # Try to locate <|im_end|> at the end
        try:
            im_end_index = sequence[::-1].index(end_token_id)
            im_end_index = len(sequence) - 1 - im_end_index
        except ValueError:
            raise ValueError("Could not find <|im_end|> token in input_ids")

        target_start = im_end_index - target_len
        labels[i, target_start:im_end_index] = input_ids[i, target_start:im_end_index]

    # Padding tokens stay ignored
    labels[input_ids == pad_token_id] = IGNORE_INDEX
    return labels

def only_assistant_response(starting_text: str, response: str) -> str:
    assert starting_text in response, f"Expected {starting_text} to be in {response}"
    # remove everything before and including the assistant token
    new_response = response.split(starting_text)[1]
    # # remove the final \n
    # new_response = new_response[:-1]
    return new_response

IGNORE_INDEX = -100  # standard for ignoring in loss



    
    # def to(
    #     self,
    #     device: torch.device = None,
    #     dtype: torch.dtype = None,
    #     non_blocking: bool = False,
    # ):
    #     if device is not None:
    #         self.model: QWenLMHeadModel = self.model.to(device=device)
    #         self.model.lm_head = self.model.lm_head.to(device=device)
    #         self.model.transformer = self.model.transformer.to(device=device)
    #         # No idea why we need to do this, shouldn't the MultiModalityCausalLM.to already do this???
    #         # print(f"moving the vision model to {device}")
    #         # self.model.vision_model = self.model.vision_model.to(device=device)
    #         # self.model.aligner = self.model.aligner.to(device=device)
    #         # self.model.language_model = self.model.language_model.to(device=device)
    #     if dtype is not None:
    #         self.model = self.model.to(dtype=dtype)
    #         self.precision_dtype = dtype

    #     return self
