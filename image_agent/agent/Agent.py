from image_agent.models.OpenAI import OpenAICaller, StructuredOpenAICaller
from image_agent.models.Qwen import VisionGeneralistCaller
from image_agent.models.Florence import VisionSpecificCaller
from image_agent.agent.AgentNodes import AgentNodes
from image_agent.agent.AgentEdges import AgentEdges
from image_agent.agent.AgentState import AgentState
from image_agent.prompts.PlanStructure import Plan, PlanStructurePrompt
from image_agent.prompts.PlanConstruction import PlanConstructionPrompt
from image_agent.prompts.ResultEvaluation import ResultAssessment, ResultEvalutionPrompt
from image_agent.agent.config import dummy_agent_config
from langgraph.graph import StateGraph, END
from langgraph.store.memory import InMemoryStore
import uuid


class Agent:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.agent_graph = self._set_up_graph()
        self.store = InMemoryStore()
        self.agent = self.agent_graph.compile(store=self.store)

    def _set_up_llm(self):
        self.planner_llm = OpenAICaller(
            api_key=self.openai_api_key, system_prompt=PlanConstructionPrompt
        )

        self.plan_structure_llm = StructuredOpenAICaller(
            api_key=self.openai_api_key,
            system_prompt=PlanStructurePrompt,
            output_model=Plan,
        )

        self.result_assessment_llm = StructuredOpenAICaller(
            api_key=self.openai_api_key,
            system_prompt=ResultEvalutionPrompt,
            output_model=ResultAssessment,
        )
        self.general_vision = VisionGeneralistCaller()
        self.specialist_vision = VisionSpecificCaller()

    def _set_up_graph(self):
        self._set_up_llm()

        nodes = AgentNodes(
            planner=self.planner_llm,
            structure=self.plan_structure_llm,
            assessment=self.result_assessment_llm,
            florence_vision=self.specialist_vision,
            qwen_vision=self.general_vision,
        )
        edges = AgentEdges()

        agent = StateGraph(AgentState)

        ## Nodes
        agent.add_node("planning", nodes.plan_node)
        agent.add_node("structure_plan", nodes.structure_plan_node)
        agent.add_node("routing", nodes.routing_node)
        agent.add_node("florence", nodes.call_florence_node)
        agent.add_node("qwen", nodes.call_qwen_node)
        agent.add_node("assessment", nodes.assessment_node)
        agent.add_node("response", nodes.dump_result_node)

        ## Edges
        agent.set_entry_point("planning")
        agent.add_edge("planning", "structure_plan")
        agent.add_edge("structure_plan", "routing")
        agent.add_conditional_edges(
            "routing",
            edges.choose_model,
            {"Florence2": "florence", "Qwen2": "qwen", "finalize": "assessment"},
        )
        agent.add_edge("florence", "routing")
        agent.add_edge("qwen", "routing")
        agent.add_conditional_edges(
            "assessment",
            edges.back_to_plan,
            {
                "good_answer": "response",
                "bad_answer": "planning",
                "timeout": "response",
            },
        )
        agent.add_edge("response", END)
        return agent

    @staticmethod
    def display_components(stage, verbose=True):
        level_1_keys = list(stage.keys())
        for k1 in level_1_keys:
            level_2_keys = list(stage[k1].keys())
            print(f"Node : {k1}")
            for k2 in level_2_keys:
                print(f"Task : {k2}")
                if verbose:
                    print(stage[k1][k2])
        print("#" * 20)

    def invoke(self, query, image, config=dummy_agent_config, max_planning_steps=2):
        user_id = config["configurable"]["user_id"]
        namespace = (user_id, "memories")
        results = []

        for i, update in enumerate(
            self.agent.stream(
                {"task": query, "image_data": image, "max_plans": max_planning_steps},
                config,
                stream_mode="updates",
            )
        ):
            # print(update)
            self.display_components(update)
            memory_id = str(uuid.uuid4())
            self.store.put(namespace, memory_id, {"memory": update})
            results.append(update)

        return results
