from mlx_vlm import load, apply_chat_template, generate
from image_agent.models.config import qwen_path
from image_agent.prompts.ImageInterpretation import ImageInterpretationPrompt


class QwenCaller:
    MODEL_PATH = qwen_path

    def __init__(self, max_tokens=1000, temperature=0):
        self.model, self.processor = load(self.MODEL_PATH)
        self.config = self.model.config
        self.max_tokens = max_tokens
        self.temperature = temperature

    def call(self, query, image):
        messages = [
            {
                "role": "system",
                "content": ImageInterpretationPrompt.system_template,
            },
            {"role": "user", "content": query},
        ]
        prompt = apply_chat_template(self.processor, self.config, messages)
        output = generate(
            self.model,
            self.processor,
            image,
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return output
