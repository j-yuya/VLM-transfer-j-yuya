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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.precision_dtype,
            trust_remote_code=True).eval().cuda()
        
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
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    ) -> torch.Tensor:
        device = self.model.device

        # Verify input image shape
        assert image.ndim == 3, f"Expected 4D image tensor, got {image.ndim}"
        #assert image.size(0) == 1, f"Expected a single image (B=1), got {image.size(0)}"

        # Repeat image for each batch sample and wrap in list for multimodal input
        B = input_ids.size(0)
        repeated_image = image.repeat(B, 1, 1, 1)  # [B, C, H, W]
        images = [[img.to(torch.bfloat16).to(device=device)] for img in repeated_image]

        # print(image)
        # print(image.shape)
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(token_type_ids.shape)
        # print(labels.shape)

        outputs = self.model(
            input_ids=input_ids.to(device),
            images=images,  # ✅ corrected
            attention_mask=attention_mask.to(device),
            token_type_ids=token_type_ids.to(device),
            labels=labels.to(device),
        )
        return outputs.loss if hasattr(outputs, "loss") else outputs[0]


    def convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        assert targets is not None, "Not support yet."
        placeholder_image = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
        
        input_by_model = self.build_inputs_with_single_image(prompts, placeholder_image)
        results = input_by_model
        print(input_by_model.keys())
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

    @torch.inference_mode()
    def generate(self, image: torch.Tensor, prompts: List[str]) -> List[str]:
        # We should only have a single image.
        
        #assert image.shape[0] == 1, print(image.shape[0])
        assert image.ndim == 3, f"Expected (1, 3, H, W), got {image.shape}"
        # we have (1, 3, h,w) , we want (3, H, W)
        model_generations = []

        for prompt in prompts:
            model_inputs = self.build_conversation_input_ids(self.tokenizer, queries=[prompt], template_version="chat")
            generated_ids = self.model.generate(
                input_ids=model_inputs["input_ids"].to(self.model.device),
                attention_mask=model_inputs["attention_mask"].to(self.model.device),
                token_type_ids=model_inputs["token_type_ids"].to(self.model.device),
                images = [[image.to(torch.bfloat16).to(self.model.device)]],
                do_sample=True if self.generation_kwargs["temperature"] > 0 else False,
                **self.generation_kwargs,
            )
            trimmed_gen_ids = generated_ids[:, model_inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(trimmed_gen_ids[0])
            model_generations.append(response)
        return model_generations
        

    def disable_model_gradients(self):
        model = self.model  # model is an instance of CogVLMForCausalLM

        model.requires_grad_(False)
        model.eval()

        # Freeze decoder (CogVLMModel)
        if hasattr(model, "model"):
            decoder = model.model
            decoder.requires_grad_(False)
            decoder.eval()

            if hasattr(decoder, "embed_tokens"):
                decoder.embed_tokens.requires_grad_(False)
                decoder.embed_tokens.eval()

            if hasattr(decoder, "layers"):
                for layer in decoder.layers:
                    layer.requires_grad_(False)
                    layer.eval()

            if hasattr(decoder, "norm"):
                decoder.norm.requires_grad_(False)
                decoder.norm.eval()

            if hasattr(decoder, "vision"):
                decoder.vision.requires_grad_(False)
                decoder.vision.eval()

        # Freeze final language modeling head
        if hasattr(model, "lm_head"):
            model.lm_head.requires_grad_(False)
            model.lm_head.eval()

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
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)
            processor = self.processor

        # Wrap prompts into chat-style message dicts
        msgs_list = []
        for i, prompt in enumerate(prompts):
            user_msg = {"role": "user", "content": [image, prompt]}
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

def only_assistant_response(starting_text: str, response: str) -> str:
    assert starting_text in response, f"Expected {starting_text} to be in {response}"
    # remove everything before and including the assistant token
    new_response = response.split(starting_text)[1]
    # # remove the final \n
    # new_response = new_response[:-1]
    return new_response

def _history_to_prompt(signal_type, history, query):
    if signal_type == 'base':
        return query
    elif signal_type == 'vqa':
        answer_format = 'Short answer:'
    elif signal_type == 'chat':
        answer_format = 'Answer:'
    else:
        assert False, f"Unknown signal type {signal_type}"

    prompt = ''
    for i, (old_query, response) in enumerate(history):
        prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += 'Question: {} {}'.format(query, answer_format)
    return prompt

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