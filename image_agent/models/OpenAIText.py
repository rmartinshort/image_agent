from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from image_agent.models.config import open_ai_model


class OpenAICaller:
    MODEL_NAME = open_ai_model

    def __init__(self, api_key, system_prompt, temperature=0, max_tokens=1000):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(
            model=self.MODEL_NAME,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
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


class StructuredOpenAICaller(OpenAICaller):
    def __init__(
        self, api_key, system_prompt, output_model, temperature=0, max_tokens=1000
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.output_model = output_model
        self.llm = ChatOpenAI(
            model=self.MODEL_NAME,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.chain = self._set_up_chain()

    def _set_up_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt.system_template),
                ("human", "{query}"),
            ]
        )
        structured_llm = self.llm.with_structured_output(self.output_model)
        chain = prompt | structured_llm

        return chain
