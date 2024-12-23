import json
from typing import Any


class AgentNodes:
    """
    A class to represent the nodes in the agent's state graph, handling various tasks related to planning,
    structuring, routing, and assessing the agent's actions.

    Attributes:
        llm_string (Any): The planner model for generating plans.
        llm_structure (Any): The model for structuring plans.
        llm_assessment (Any): The model for assessing plans.
        florence (Any): The vision model for specialized tasks.
        qwen (Any): The vision model for general tasks.
    """

    def __init__(
        self,
        planner: Any,
        structure: Any,
        assessment: Any,
        special_vision: Any,
        general_vision: Any,
    ) -> None:
        """
        Initializes the AgentNodes with the specified models for planning, structuring, assessing, and vision.

        Args:
            planner (Any): The planner model for generating plans.
            structure (Any): The model for structuring plans.
            assessment (Any): The model for assessing plans.
            florence_vision (Any): The vision model for specialized tasks.
            qwen_vision (Any): The vision model for general tasks.
        """
        self.llm_string: Any = planner
        self.llm_structure: Any = structure
        self.llm_assessment: Any = assessment
        self.special_vision: Any = special_vision
        self.general_vision: Any = general_vision

    def plan_node(self, state: dict) -> dict:
        """
        Generates a new plan based on the current task and previous responses.

        Args:
            state (dict): The current state of the agent, containing task and previous plan information.

        Returns:
            dict: A dictionary containing the new plan and the updated plan version.
        """
        agent_task = state["task"]
        plan_version = state.get("plan_version", 0)
        previous_response = state.get("answer_assessment", None)
        previous_plan = state.get("plan", None)

        if previous_plan and previous_response:
            input_task = f"The task is {agent_task}\nYour old plan was {previous_plan} \n but your answer wasn't good enough. Another system provided this feedback: \n {previous_response}. \n Please revise your plan"
        else:
            input_task = f"The task is {agent_task}"

        response = self.llm_string.call(input_task)
        return {"plan": response, "plan_version": plan_version + 1}

    @staticmethod
    def post_process_plan_structure(plan_structure: Any) -> dict:
        """
        Processes the plan structure to create a structured plan.

        Args:
            plan_structure (Any): The plan structure to process.

        Returns:
            dict: A structured plan with steps indexed starting from 1.
        """
        final_plan = plan_structure.model_dump()["plan"]
        structured_plan = {}
        for i, step in enumerate(final_plan):
            structured_plan[i + 1] = step
        return structured_plan

    def structure_plan_node(self, state: dict) -> dict:
        """
        Structures the generated plan and prepares it for execution.

        Args:
            state (dict): The current state of the agent, containing the generated plan.

        Returns:
            dict: A dictionary containing the structured plan and step information.
        """
        messages = state["plan"]
        response = self.llm_structure.call(messages)
        final_plan_dict = self.post_process_plan_structure(response)
        final_plan = json.dumps(final_plan_dict)

        return {
            "plan_structure": final_plan,
            "current_step": 0,
            "max_steps": len(final_plan_dict),
        }

    def routing_node(self, state: dict) -> dict:
        """
        Updates the current step in the routing process.

        Args:
            state (dict): The current state of the agent.

        Returns:
            dict: A dictionary containing the updated current step.
        """
        plan_stage = state.get("current_step", 0)
        return {"current_step": plan_stage + 1}

    def call_special_vision_node(self, state: dict) -> dict:
        """
        Calls the specialized vision model with the current step's input.

        Args:
            state (dict): The current state of the agent, containing the plan structure and image data.

        Returns:
            dict: A dictionary containing the output from the specialized vision model.
        """
        plan_stage = state.get("current_step")
        florence_input = json.loads(state.get("plan_structure"))[str(plan_stage)]

        florence_mode = florence_input["tool_mode"]
        florence_text = florence_input["tool_input"]

        if florence_text and len(florence_text) < 1:
            florence_text = None

        florence_output = self.special_vision.call(
            task_prompt=florence_mode,
            image=state.get("image_data"),
            text_input=florence_text,
        )
        return {
            "plan_output": [{plan_stage: json.dumps(florence_output)}],
        }

    def call_general_vision_node(self, state: dict) -> dict:
        """
        Calls the general vision model with the current step's input.

        Args:
            state (dict): The current state of the agent, containing the plan structure and image data.

        Returns:
            dict: A dictionary containing the output from the general vision model.
        """
        plan_stage = state.get("current_step")
        qwen_input = json.loads(state.get("plan_structure"))[str(plan_stage)]

        qwen_text = qwen_input["tool_input"]
        qwen_output = self.general_vision.call(
            query=qwen_text, image=state.get("image_data")
        )
        return {
            "plan_output": [{plan_stage: str(qwen_output)}],
        }

    def assessment_node(self, state: dict) -> dict:
        """
        Assesses the generated plan and output based on the user question.

        Args:
            state (dict): The current state of the agent, containing the user question and plan information.

        Returns:
            dict: A dictionary containing the assessment of the answer and a flag indicating the result.
        """
        user_question = state.get("task")
        model_plan = state.get("plan")
        output_so_far = str(state.get("plan_output", []))
        llm_input = f"The question was: {user_question} \nThe plan was:\n {model_plan}\nThe output is:\n {output_so_far}"
        response = self.llm_assessment.call(llm_input).model_dump()

        return {
            "answer_assessment": response["assessment"],
            "answer_flag": response["final_answer"],
        }

    def dump_result_node(self, state: dict) -> dict:
        """
        Dumps the final result of the assessment and outputs.

        Args:
            state (dict): The current state of the agent, containing the output and assessment.

        Returns:
            dict: A dictionary containing the final assessment and output results.
        """
        output_so_far = state.get("plan_output", [])
        final_response = state.get("answer_assessment", "")
        return {"answer_assessment": final_response, "final_result": output_so_far}
