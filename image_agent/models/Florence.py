from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
import os
from image_agent.models.config import florence_path
from typing import Optional, Any


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
    """
    A class to interact with the Florence model for various vision-language tasks.

    Attributes:
        MODEL_PATH (str): Path to the pre-trained Florence model.
        TASK_DICT (dict): A dictionary mapping task names to task codes.
    """

    MODEL_PATH: str = florence_path  # Replace `florence_path` with the actual path or variable definition.
    TASK_DICT: dict[str, str] = {
        "general object detection": "<OD>",
        "specific object detection": "<CAPTION_TO_PHRASE_GROUNDING>",
        "image captioning": "<MORE_DETAILED_CAPTION>",
        "OCR": "<OCR_WITH_REGION>",
    }

    def __init__(self) -> None:
        """
        Initializes the FlorenceCaller instance by loading the model and processor.

        The model and processor are loaded using the specified `MODEL_PATH` and moved to the appropriate device.
        """
        self.device: str = (
            get_device_type()
        )  # Function to determine the device type (e.g., 'cpu' or 'cuda').

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                self.MODEL_PATH, trust_remote_code=True
            )
            self.processor: AutoProcessor = AutoProcessor.from_pretrained(
                self.MODEL_PATH, trust_remote_code=True
            )
            self.model.to(self.device)

    def translate_task(self, task_name: str) -> str:
        """
        Translates a human-readable task name into its corresponding task code.

        Args:
            task_name (str): The name of the task (e.g., "general object detection").

        Returns:
            str: The corresponding task code. Defaults to "<DETAILED_CAPTION>" if the task name is not found.
        """
        return self.TASK_DICT.get(task_name, "<DETAILED_CAPTION>")

    def call(
        self, task_prompt: str, image: Any, text_input: Optional[str] = None
    ) -> Any:
        """
        Executes a vision-language task using the Florence model.

        Args:
            task_prompt (str): The name of the task to perform (e.g., "image captioning").
            image (Any): The input image for the task (e.g., a PIL Image object).
            text_input (Optional[str]): Additional text input for tasks that require it. Defaults to None.

        Returns:
            Any: The parsed output of the task as processed by the Florence model.
        """
        # Get the corresponding task code for the given prompt
        task_code: str = self.translate_task(task_prompt)

        # Prevent text_input for tasks that do not require it
        if task_code in [
            "<OD>",
            "<MORE_DETAILED_CAPTION>",
            "<OCR_WITH_REGION>",
            "<DETAILED_CAPTION>",
        ]:
            text_input = None

        # Construct the prompt based on whether text_input is provided
        prompt: str = task_code if text_input is None else task_code + text_input

        # Preprocess inputs for the model
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device
        )

        # Generate predictions using the model
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        # Decode and process generated output
        generated_text: str = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer: dict[str, Any] = self.processor.post_process_generation(
            generated_text, task=task_code, image_size=(image.width, image.height)
        )

        return parsed_answer[task_code]
