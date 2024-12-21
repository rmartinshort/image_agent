import json


class AgentEdges:
    @staticmethod
    def choose_model(state):
        current_plan = json.loads(state.get("plan_structure"))
        current_step = state.get("current_step", 1)
        max_step = state.get("max_steps", 999)

        if current_step > max_step:
            return "finalize"
        else:
            step_to_execute = current_plan[str(current_step)]["tool_name"]
            return step_to_execute

    @staticmethod
    def back_to_plan(state):
        assessment_flag = state.get("answer_flag", 0)
        iteration_number = state.get("plan_version", 0)
        max_plans = state.get("max_plans", 1)

        if assessment_flag:
            return "good_answer"
        elif iteration_number > max_plans:
            return "timeout"
        else:
            return "bad_answer"
