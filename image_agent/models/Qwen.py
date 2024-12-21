from mlx_vlm import load, apply_chat_template, generate
from image_agent.models.config import qwen_path


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
                "content": """
            You are a helpful assistant who answers open-ended questions about images.
            Keep your responses relevant and concise, fewer than 50 words.
            You will recieve a prompt in english and you MUST also respond in english.
            """,
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
