from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
import os
from image_agent.models.config import florence_path


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def get_device_type():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    else:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        else:
            return "cpu"


class FlorenceCaller:
    MODEL_PATH = florence_path
    TASK_DICT = {
        "general object detection": "<OD>",
        "specific object detection": "<CAPTION_TO_PHRASE_GROUNDING>",
        "image captioning": "<MORE_DETAILED_CAPTION>",
        "OCR": "<OCR_WITH_REGION>",
    }

    def __init__(self):
        self.device = get_device_type()

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_PATH, trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_PATH, trust_remote_code=True
            )
            self.model.to(self.device)

    def translate_task(self, task_name):
        task_code = self.TASK_DICT.get(task_name, "<DETAILED_CAPTION>")
        return task_code

    def call(self, task_prompt, image, text_input=None):
        task_code = self.translate_task(task_prompt)

        # prevent bug where previous model gives text_input where it shouldn't belong
        if task_code in [
            "<OD>",
            "<MORE_DETAILED_CAPTION>",
            "<OCR WITH REGION>",
            "<DETAILED_CAPTION>",
        ]:
            text_input = None

        if text_input is None:
            prompt = task_code
        else:
            prompt = task_code + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device
        )
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_code, image_size=(image.width, image.height)
        )
        return parsed_answer[task_code]
