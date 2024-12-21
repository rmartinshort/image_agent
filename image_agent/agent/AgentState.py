from typing_extensions import TypedDict
from IPython.display import Image
from typing import List, Dict, Annotated
from operator import add


class AgentState(TypedDict):
    """ """

    task: str
    plan: str
    plan_version: int
    max_plans: int
    image_data: Image
    plan_structure: str
    current_step: int
    max_steps: int
    plan_output: Annotated[List[Dict[int, str]], add]
    answer_assessment: str
    answer_flag: int
    final_result: List[str]
