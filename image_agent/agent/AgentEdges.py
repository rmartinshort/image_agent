import json


class AgentEdges:
    """
    A class to define the edges for the agent's state graph, providing methods to determine transitions
    based on the current state of the agent.

    Methods:
        choose_model(state: dict) -> str:
            Determines the next model to execute based on the current plan and step.

        back_to_plan(state: dict) -> str:
            Determines the next action based on the assessment of the current answer and iteration.
    """

    @staticmethod
    def choose_model(state: dict) -> str:
        """
        Determines the next model to execute based on the current plan and step.

        Args:
            state (dict): The current state of the agent, containing plan structure and step information.

        Returns:
            str: The name of the tool to execute next, or "finalize" if the maximum step is exceeded.
        """
        current_plan = json.loads(state.get("plan_structure"))
        current_step = state.get("current_step", 1)
        max_step = state.get("max_steps", 999)

        if current_step > max_step:
            return "finalize"
        else:
            step_to_execute = current_plan[str(current_step)]["tool_name"]
            return step_to_execute

    @staticmethod
    def back_to_plan(state: dict) -> str:
        """
        Determines the next action based on the assessment of the current answer and iteration.

        Args:
            state (dict): The current state of the agent, containing assessment flags and iteration numbers.

        Returns:
            str: The next action to take, which can be "good_answer", "timeout", or "bad_answer".
        """
        assessment_flag = state.get("answer_flag", 0)
        iteration_number = state.get("plan_version", 0)
        max_plans = state.get("max_plans", 1)

        if assessment_flag:
            return "good_answer"
        elif iteration_number > max_plans:
            return "timeout"
        else:
            return "bad_answer"
