from image_agent.models.OpenAIText import OpenAICaller, StructuredOpenAICaller
from image_agent.models.Qwen import QwenCaller
from image_agent.models.Florence import FlorenceCaller
from image_agent.models.OpenAIVision import OpenAIVisionCaller
from image_agent.agent.AgentNodes import AgentNodes
from image_agent.agent.AgentEdges import AgentEdges
from image_agent.agent.AgentState import AgentState
from image_agent.prompts.PlanStructure import Plan, PlanStructurePrompt
from image_agent.prompts.PlanConstruction import PlanConstructionPrompt
from image_agent.prompts.ResultEvaluation import ResultAssessment, ResultEvalutionPrompt
from image_agent.prompts.ImageInterpretation import ImageInterpretationPrompt
from image_agent.agent.config import dummy_agent_config
from langgraph.graph import StateGraph, END
from langgraph.store.memory import InMemoryStore
import uuid
from typing import Any
import logging


logging.basicConfig(
    format="%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("Agent")
logger.setLevel(logging.INFO)


class Agent:
    """
    A class to represent an AI agent that interacts with OpenAI's API and manages a state graph.

    Attributes:
        openai_api_key (str): The API key for OpenAI.
        agent_graph (StateGraph): The state graph representing the agent's workflow.
        store (InMemoryStore): The in-memory store for agent's data.
        agent (StateGraph): The compiled agent graph.
    """

    def __init__(self, openai_api_key: str, vision_mode="local"):
        """
        Initializes the Agent with the provided OpenAI API key.

        Args:
            openai_api_key (str): The API key for OpenAI.
        """
        self.openai_api_key: str = openai_api_key
        self.vision_mode = vision_mode
        self.agent_graph: StateGraph = self._set_up_graph()
        self.store: InMemoryStore = InMemoryStore()
        self.agent = self.agent_graph.compile(store=self.store)

    def _set_up_llm(self) -> None:
        """
        Sets up the various language models used by the agent.
        """
        self.planner_llm: OpenAICaller = OpenAICaller(
            api_key=self.openai_api_key, system_prompt=PlanConstructionPrompt
        )

        self.plan_structure_llm: StructuredOpenAICaller = StructuredOpenAICaller(
            api_key=self.openai_api_key,
            system_prompt=PlanStructurePrompt,
            output_model=Plan,
        )

        self.result_assessment_llm: StructuredOpenAICaller = StructuredOpenAICaller(
            api_key=self.openai_api_key,
            system_prompt=ResultEvalutionPrompt,
            output_model=ResultAssessment,
        )

        logger.info(f"General vision mode is {self.vision_mode}")
        if self.vision_mode == "local":
            self.general_vision: QwenCaller = QwenCaller()
        elif self.vision_mode == "gpt":
            self.general_vision: OpenAIVisionCaller = OpenAIVisionCaller(
                api_key=self.openai_api_key, system_prompt=ImageInterpretationPrompt
            )
        else:
            raise ValueError("Vision mode must be local or gpt")

        self.specialist_vision: FlorenceCaller = FlorenceCaller()

    def _set_up_graph(self) -> StateGraph:
        """
        Sets up the state graph for the agent.

        Returns:
            StateGraph: The configured state graph.
        """
        self._set_up_llm()

        nodes: AgentNodes = AgentNodes(
            planner=self.planner_llm,
            structure=self.plan_structure_llm,
            assessment=self.result_assessment_llm,
            special_vision=self.specialist_vision,
            general_vision=self.general_vision,
        )
        edges: AgentEdges = AgentEdges()

        agent: StateGraph = StateGraph(AgentState)

        ## Nodes
        agent.add_node("planning", nodes.plan_node)
        agent.add_node("structure_plan", nodes.structure_plan_node)
        agent.add_node("routing", nodes.routing_node)
        agent.add_node("special_vision", nodes.call_special_vision_node)
        agent.add_node("general_vision", nodes.call_general_vision_node)
        agent.add_node("assessment", nodes.assessment_node)
        agent.add_node("response", nodes.dump_result_node)

        ## Edges
        agent.set_entry_point("planning")
        agent.add_edge("planning", "structure_plan")
        agent.add_edge("structure_plan", "routing")
        agent.add_conditional_edges(
            "routing",
            edges.choose_model,
            {
                "special_vision": "special_vision",
                "general_vision": "general_vision",
                "finalize": "assessment",
            },
        )
        agent.add_edge("special_vision", "routing")
        agent.add_edge("general_vision", "routing")
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
    def display_components(stage: dict, verbose: bool = True) -> None:
        """
        Displays the components of the given stage.

        Args:
            stage (dict): The stage containing nodes and tasks.
            verbose (bool): Flag to control verbosity of output.
        """
        level_1_keys = list(stage.keys())
        for k1 in level_1_keys:
            level_2_keys = list(stage[k1].keys())
            print(f"Node : {k1}")
            for k2 in level_2_keys:
                print(f"Task : {k2}")
                if verbose:
                    print(stage[k1][k2])
        print("#" * 20)

    def invoke(
        self,
        query: str,
        image: Any,
        config: dict = dummy_agent_config,
        max_planning_steps: int = 2,
    ) -> list:
        """
        Invokes the agent with a query and an image, returning the results.

        Args:
            query (str): The query to process.
            image (Any): The image data associated with the query.
            config (dict): Configuration options for the agent.
            max_planning_steps (int): The maximum number of planning steps to execute.

        Returns:
            list: The results generated by the agent.
        """
        user_id: str = config["configurable"]["user_id"]
        namespace: tuple = (user_id, "memories")
        results: list = []

        for i, update in enumerate(
            self.agent.stream(
                {"task": query, "image_data": image, "max_plans": max_planning_steps},
                config,
                stream_mode="updates",
            )
        ):
            logger.info(f"At agent step {i}")
            self.display_components(update)
            memory_id: str = str(uuid.uuid4())
            self.store.put(namespace, memory_id, {"memory": update})
            results.append(update)

        return results
