from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langsmith import Client
import os


llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=1000 
)


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


search_tool = TavilySearchResults(search_depth="basic")

tools = [get_system_time, search_tool]

client = Client()
react_prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)