# image_agent
Use LanGraph, OpenAI, MLX and HuggingFace tools to create a basic multimodal AI agent that can answer multi-step questions about an image.

To test it, provide an image and an associated query. Then take a look at the code in `example.py`.

First make a new Python 3.10+ environment and install the requirements here.

Then you can test with a minimal testing like this:
```python

# load all the relevant libraries
from image_agent.agent.Agent import Agent
from PIL import Image
from image_agent.utils import load_secrets

# make sure you have a .env file in this top level directory with 
# the line OPENAI_API_KEY = "{your key}"
secrets = load_secrets()

# path to your image
loaded_image_path = "dogs.jpg"
loaded_image = Image.open(loaded_image_path)

query = "Find all the dogs in this image and tell me what each one is doing. Also, what's the weather like?"

# if vision_mode = "gpt", we will use GPT4o-mini for the generalist vision task
# if vision_mode = "local", it will use QWEN2 VL with MLX for general vision tasks
# this will first download the model from HuggingFace if it is not already present on
# your device
agent = Agent(openai_api_key=secrets["OPENAI_API_KEY"],vision_mode="gpt")

# result will be a list containing the outputs of all the agent steps
result = agent.invoke(query, loaded_image)
```
