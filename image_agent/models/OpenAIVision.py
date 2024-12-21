from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from image_agent.models.config import open_ai_model
from image_agent.image_tools import convert_PIL_to_base64, resize_maintain_aspect


class OpenAIVisionCaller:
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

    def call(self, query, image, standard_width=512):
        image = resize_maintain_aspect(image, standard_width)
        base64image = convert_PIL_to_base64(image)

        human_msg = [
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64image}"},
            {"type": "text", "text": query},
        ]

        return self.chain.invoke({"query": human_msg})
