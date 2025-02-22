from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Optional, List


class PlanComponent(BaseModel):
    """Information about one stage of the plan"""

    tool_name: str = Field(
        description="The name of the tool to be called. Must be either special_vision or general_vision"
    )
    tool_mode: str = Field(
        description="The mode inwhich to call the tool. Must be chosen from the list given in the prompt"
    )
    tool_input: Optional[str] = Field(description="The input text for the tool")


class Plan(BaseModel):
    """The entire plan as a list of steps"""

    plan: List[PlanComponent] = Field(description="The plan")


@dataclass
class PlanStructurePrompt:
    system_template: str = """
    Your task is to take a text description of a plan and convert it into a structured output
    The plan will consist of a list of components. Each component will have the following:
    1. The name of the tool
    2. The name of the mode inwhich the tool is called 
    3. A string input, if necessary

    The tool name be chosen from the following:
    - special_vision (for finding bounding boxes, doing OCR other specialist vision tasks where numerical outputs are needed)
    - general_vision (for answering general image questions where text rather than numerical outputs are not needed)

    The tool mode must be chosen from the following:
    "general object detection".
    - This is for general object detection, call when we need to detect all objects and are not told their specific names
    "specific object detection". 
    - This is for specific object detection, call when we need to detect a specific named object
    "image captioning".
    - This is for image captioning, call when we are asked to provide a caption
    "OCR".
    - OCR, call when we want to extract text
    "conversation".
    - General conversation, call when we are asked a more complex question about the image 
    Please choose ONLY from the list above when selecting tool mode. Return the exact names of the tool modes you chose

    Please study the following rules before crafting your response:

    If tool name = special_vision and tool mode = 'specific object detection', then tool_input must be provided 
    If tool name = general_vision then tool model must be 'conversation' and tool_input must be provided

    Here is an example:

    Input: "call specialist vision to detect all objects. Then call specialist vision in specific mode with input "cats". Then call generalist vision to describe the image"
    You would choose three tools:
    1. special_vision in 'general object detection' mode with tool_input = None
    2. special_vision in 'specific object detection' with tool_input = "cat"
    3. general_vision in 'conversation' with tool_input = "describe this image"
    Note that you have used exact names from the lists of options above. You must always do this, regardless of what the input is.
    """
