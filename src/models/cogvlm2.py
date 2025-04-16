# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
from src.models.label_compute import make_labels
from src.models.qwen_utils.modeling_qwen import QWenLMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import lightning
import torch
from typing import Any, Callable, Dict, List, Mapping, Optional, Literal, Tuple
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
from src.models.base import VisionLanguageModel
from PIL import Image
from torchvision import transforms

from src.models.qwen_utils.visual import VisionTransformer
LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

# Labels with these indices will be ignored by cross entropy loss in PyTorch.
IGNORE_INDEX = -100


class CogVLM2(VisionLanguageModel, lightning.LightningModule):
    def __init__(
        self,
        #TODO: More Param Variants
        model_str: str = "cogvlm2-llama3-chat-19B",
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

        model_path = f"THUDM/{model_str}"

        # not sure why we need to register the image processor manually
        print(f"Using CogVLM2 model: {model_path}")

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
        assert image.ndim == 3, f"Expected 3D image tensor, got {image.ndim}"
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

        input_by_model = self.build_conversation_input_ids(
            self.tokenizer,
            queries=prompts,
            template_version='chat',
            answers=targets,
        )
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
            response = response.split("<|end_of_text|>")[0]
            model_generations.append(response)
        return model_generations
        

    def disable_model_gradients(self):
        # Turn off gradients everywhere under self.model
        self.model.requires_grad_(False)
        # Switch everything to eval mode (no dropout etc.)
        self.model.eval()

    def build_conversation_input_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        *,
        queries: List[str],
        answers: Optional[List[str]] = None,
        history: Optional[List[Tuple[str, str]]] = None,
        template_version: Optional[Literal["base", "chat", "vqa"]] = None,
    ):
        image_size: int = self.model.config.vision_config['image_size']
        patch_size: int = self.model.config.vision_config['patch_size']
        template_version = template_version or self.config.template_version
        tokenizer.pad_token_id = 128002  # llama3 adapt for cogvlm

        # Token counts for vision tokens (from model arch spec)
        vision_token_num = (image_size // patch_size // 2) * (image_size // patch_size // 2) + 2

        history = history or []

        input_ids_list = []
        token_type_ids_list = []
        attention_masks = []
        labels_list = []

        for i, query in enumerate(queries):
            text = _history_to_prompt(template_version, history, query)
            input_ids = [tokenizer.bos_token_id] + [tokenizer.pad_token_id] * vision_token_num
            token_type_ids = [LANGUAGE_TOKEN_TYPE] + [VISION_TOKEN_TYPE] * vision_token_num

            text_ids = tokenizer.encode(text, add_special_tokens=False)

            if answers is not None and i < len(answers):
                answer_ids = tokenizer.encode(answers[i], add_special_tokens=False) + [tokenizer.eos_token_id]
                full_ids = text_ids + answer_ids
            else:
                answer_ids = None
                full_ids = text_ids

            input_ids += full_ids
            token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(full_ids)

            attention_mask = [1] * len(input_ids)

            if answer_ids is not None:
                labels = [-100] * (len(input_ids) - len(answer_ids)) + answer_ids
            else:
                labels = None

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            token_type_ids_list.append(torch.tensor(token_type_ids, dtype=torch.long))
            attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
            if labels is not None:
                labels_list.append(torch.tensor(labels, dtype=torch.long))

        # Pad all sequences to the max length (left-padding)
        def pad_to_max(seq_list, pad_value):
            max_len = max(seq.size(0) for seq in seq_list)
            padded = [torch.cat([torch.full((max_len - seq.size(0),), pad_value, dtype=seq.dtype), seq]) for seq in seq_list]
            return torch.stack(padded)

        input_ids = pad_to_max(input_ids_list, tokenizer.pad_token_id)
        token_type_ids = pad_to_max(token_type_ids_list, LANGUAGE_TOKEN_TYPE)
        attention_mask = pad_to_max(attention_masks, 0)
        labels = pad_to_max(labels_list, -100) if labels_list else None

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

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