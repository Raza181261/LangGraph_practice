import datetime
import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schema import AnswerQuestion
from schema import RevisedAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import AIMessage, BaseMessage,HumanMessage
from dotenv import load_dotenv


pydantic_parser = PydanticToolsParser(tools = [AnswerQuestion])

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=1000 
)

actor_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a expert AI researcher.
        Current time: {time}

        1. {first_instruction}
        2. Reflect and critique your answer. Be server to maximize improvement.
        3. After the reflection, **list 1-3 search queries separatly** for researching improvements. Do not include them inside the reflection.
        """,
        ),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Answer the user's question above using the required format.")
    
]).partial(
    time = lambda: datetime.datetime.now().isoformat()
)


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Answer the user's question in 250 words."
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion")

valildator = PydanticToolsParser(tools=[AnswerQuestion])


# Reviser section
reviser_instructions = """"
  Revise your previous answer using the new information.
You should use the previous critique to add important information to your answer.
    - You must include numerical statistics to ensure it is verifiable.
    - You should use the previous critique to remove superfluous information that does not make the answer more verifiable.
    - You must include a "References" section at the bottom of your answer (which does not count towards the word limit). In the form of:
        - [1] https://example.com
        - [2] https://example.com
You should make sure that the answer is not more than 250 words.
"""

revisor_chain = actor_prompt_template.partial(
    first_instruction = reviser_instructions
) | llm.bind_tools(
    tools=[RevisedAnswer], tool_choice="RevisedAnswer")

response =  first_responder_chain.invoke({
    "messages": [HumanMessage(content="write me blog post on how small businesses can leverage AI to improve their operations.")],
})

print("First Response:", response)