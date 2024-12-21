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

    query = (
        "Describe the weather and then find the white dog. What is the white dog doing?"
    )

    agent = Agent(openai_api_key=secrets["OPENAI_API_KEY"])
    result = agent.invoke(query, loaded_image)

    for component in result:
        print(component)
        print("\n---------------------\n")
