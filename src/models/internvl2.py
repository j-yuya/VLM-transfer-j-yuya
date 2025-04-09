# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
from src.models.label_compute import make_labels
from src.models.qwen_utils.modeling_qwen import QWenLMHeadModel
from transformers import AutoTokenizer, AutoModel
import lightning
import torch
from typing import Any, Callable, Dict, List, Mapping, Optional
from transformers import PreTrainedTokenizer

from src.models.base import VisionLanguageModel


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
        precision: str = "bf16-mixed",
    ):
        super().__init__()
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
        import pdb
        # Since we only get a single image, we need to repeat it for the batch size.
        assert image.ndim == 5, f"Expected 4 dims, got {image.ndim}"
        # assert that we only have one image here
        assert (
            image.size(0) == 1
        ), f"Expected only 1 image that we repeat, got {image.size(0)}"
        image = image.repeat(1, 1, 1, 1, 1)
        image_flags = torch.tensor([1] * image.shape[1], dtype=torch.long)
        pdb.set_trace()
        outputs = self.model(
            input_ids=input_ids.to(device=device),
            pixel_values=image.squeeze(0).to(torch.bfloat16).to(device=device),
            attention_mask=attention_mask.to(device=device),
            image_flags = image_flags.to(device=device),
            labels=labels.to(device=device),
        )
        
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
        num_patches_list = [torch.tensor([5]) for _ in prompts]
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        prompt_texts = []

        for prompt, target, num_patches in zip(prompts, targets, num_patches_list):
            # Construct image placeholder with visual tokens
            prompt_w_image_tag = f"<image>\n{prompt}"
            query = format_instruction_internvl(prompt_w_image_tag)
            for num_patches in num_patches_list:
                visual_token_str = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * NUM_IMAGE_TOKENS * num_patches) + IMG_END_TOKEN
                query = query.replace('<image>', visual_token_str, 1)
                if target is not None:
                    query = query + target
            #image_plus_prompt = f"{visual_token_str}\n{prompt}"

            prompt_texts.append(query)
        
        model_inputs = self.tokenizer(prompt_texts, padding=True, truncation=False, return_tensors='pt')
        results = {}
        results["input_ids"] = model_inputs.input_ids
        results["attention_mask"] = model_inputs.attention_mask



        
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
            # first_text = prompt_texts[0]
            # print(f"First text: {first_text}")
            print(f"First input_ids: {input_ids[0]}")
            print(f"First attention_mask: {attention_mask[0]}")
            print(f"First labels: {results['labels'][0]}")
            if len(input_ids) > 1:
                print(f"Second input ids: {input_ids[1]}")
                print(f"Second attention_mask: {attention_mask[1]}")
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
        num_patches_list = [torch.tensor([5]) for _ in prompts]
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
            import pdb
            pdb.set_trace()
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

def only_assistant_response(starting_text: str, response: str) -> str:
    assert starting_text in response, f"Expected {starting_text} to be in {response}"
    # remove everything before and including the assistant token
    new_response = response.split(starting_text)[1]
    # # remove the final \n
    # new_response = new_response[:-1]
    return new_response

    # def disable_model_gradients(self):
    #     self.model.requires_grad_(False)
    #     self.model.eval()
    #     self.model.transformer.requires_grad_(False)
    #     self.model.transformer.eval()
    #     self.vision_model.requires_grad_(False)
    #     self.vision_model.eval()

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
