from image_agent.agent.Agent import Agent
from PIL import Image
from image_agent.utils import load_secrets
import os


def main():
    secrets = load_secrets()
    example_image = "dogs.jpg"
    loaded_image_path = os.path.join(
        os.path.dirname(__file__), "example_images", example_image
    )
    loaded_image = Image.open(loaded_image_path)

    # good example to illustrate limitation of local model
    # this will give the correct answer with GPT4 but with Qwen it does not
    query = (
        "What are the dogs in this image doing? Find the white dog and tell me if its facing the camera"
    )

    agent = Agent(openai_api_key=secrets["OPENAI_API_KEY"],vision_mode="gpt")
    result = agent.invoke(query, loaded_image)

    for component in result:
        print(component)
        print("\n---------------------\n")
