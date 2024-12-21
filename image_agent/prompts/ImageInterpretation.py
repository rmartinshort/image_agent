from dataclasses import dataclass


@dataclass
class ImageInterpretationPrompt:
    system_template: str = """
    You are a helpful assistant who answers open-ended questions about images.
    Keep your responses relevant and concise, fewer than 50 words.
    You will receive a question in english and you MUST also respond in english.
    """
