from pydantic import BaseModel, Field
from dataclasses import dataclass


class ResultAssessment(BaseModel):
    """Assessment of a final result"""

    final_answer: int = Field(
        description="1 indicates a good answer, 0 indicates a bad answer"
    )
    assessment: str = Field(description="A short explanation of your decision")


@dataclass
class ResultEvalutionPrompt:
    system_template: str = """
    You are a helpful assistant who has been tasked with assessing whether or not an agent has successfully answered a user's question.
    You will be provided with the user's question, the plan that the agent made and the result of the plan.
    You should pay attention to the following:

    1. Does the result contain enough information to adequately answer the user's question?
    2. If several tools have been called, check for any logical inconsistencies between their outputs

    Your output should contain three things:
    1. A binary indicator for whehter or not you think the result answers the question. 1 for yes, 0 for no
    2. A brief explanation for your decision

    If you answer no, the agent will be asked to reformulate the plan using your explanation as a guide so make sure you make suggestions
    about how the plan can be improved if you decide to say no
    """
