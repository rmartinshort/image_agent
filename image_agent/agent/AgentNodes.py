import json


class AgentNodes:
    def __init__(self, planner, structure, assessment, florence_vision, qwen_vision):
        self.llm_string = planner
        self.llm_structure = structure
        self.llm_assessment = assessment
        self.florence = florence_vision
        self.qwen = qwen_vision

    def plan_node(self, state):
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
    def post_process_plan_structure(plan_structure):
        final_plan = plan_structure.model_dump()["plan"]
        structured_plan = {}
        for i, step in enumerate(final_plan):
            structured_plan[i + 1] = step
        return structured_plan

    def structure_plan_node(self, state):
        messages = state["plan"]
        response = self.llm_structure.call(messages)
        final_plan_dict = self.post_process_plan_structure(response)
        final_plan = json.dumps(final_plan_dict)

        return {
            "plan_structure": final_plan,
            "current_step": 0,
            "max_steps": len(final_plan_dict),
        }

    def routing_node(self, state):
        plan_stage = state.get("current_step", 0)
        return {"current_step": plan_stage + 1}

    def call_florence_node(self, state):
        plan_stage = state.get("current_step")

        florence_input = json.loads(state.get("plan_structure"))[str(plan_stage)]

        florence_mode = florence_input["tool_mode"]
        florence_text = florence_input["tool_input"]

        if florence_text and len(florence_text) < 1:
            florence_text = None

        florence_output = self.florence.call(
            task_prompt=florence_mode,
            image=state.get("image_data"),
            text_input=florence_text,
        )
        return {
            "plan_output": [{plan_stage: json.dumps(florence_output)}],
        }

    def call_qwen_node(self, state):
        plan_stage = state.get("current_step")
        qwen_input = json.loads(state.get("plan_structure"))[str(plan_stage)]

        qwen_text = qwen_input["tool_input"]
        qwen_output = self.qwen.call(query=qwen_text, image=state.get("image_data"))
        return {
            "plan_output": [{plan_stage: str(qwen_output)}],
        }

    def assessment_node(self, state):
        user_question = state.get("task")
        model_plan = state.get("plan")
        output_so_far = str(state.get("plan_output", []))
        llm_input = f"The question was: {user_question} \nThe plan was:\n {model_plan}\nThe output is:\n {output_so_far}"
        response = self.llm_assessment.call(llm_input).model_dump()

        return {
            "answer_assessment": response["assessment"],
            "answer_flag": response["final_answer"],
        }

    def dump_result_node(self, state):
        output_so_far = state.get("plan_output", [])
        final_response = state.get("answer_assessment", "")
        return {"answer_assessment": final_response, "final_result": output_so_far}
