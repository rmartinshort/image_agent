from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX
from image_agent.models.config import llama_path


class LlamaCaller:
    MODEL_PATH = llama_path

    def __init__(self, system_prompt, temperature=0, max_tokens=1000):
        self.system_prompt = system_prompt
        self.loaded_model = MLXPipeline.from_model_id(
            self.MODEL_PATH,
            pipeline_kwargs={
                "max_tokens": max_tokens,
                "temp": temperature,
                "do_sample": False,
            },
        )
        self.llm = ChatMLX(llm=self.loaded_model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chain = self._set_up_chain()

    def _set_up_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt.system_template),
                ("human", "{query}"),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain

    def call(self, query):
        return self.chain.invoke({"query": query})


class StructuredLlamaCaller(LlamaCaller):
    def __init__(self, system_prompt, output_model, temperature=0, max_tokens=1000):
        super().__init__(system_prompt, temperature, max_tokens)
        self.system_prompt = system_prompt
        self.output_model = output_model
        self.loaded_model = MLXPipeline.from_model_id(
            self.MODEL_PATH,
            pipeline_kwargs={
                "max_tokens": max_tokens,
                "temp": temperature,
                "do_sample": False,
            },
        )
        self.llm = ChatMLX(llm=self.loaded_model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chain = self._set_up_chain()

    def _set_up_chain(self):
        # Set up a parser
        parser = PydanticOutputParser(pydantic_object=self.output_model)

        # Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.system_prompt.system_template,
                ),
                ("human", "{query}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.llm | parser
        return chain
