# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from langchain.agents import initialize_agent
# from langchain_community.tools import TavilySearchResults

# load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2, max_output_tokens=100)

# search_tool = TavilySearchResults(search_depth = "basic")

# tools = [search_tool]

# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent="zero-shot-react-description",
#     verbose=True
# )

# agent.invoke("Give me a news about LAhore weather today")



# # result = llm.invoke("Write a short poem about the beauty of nature.")

# # print(result)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_output_tokens=100)

search_tool = TavilySearch(max_results=5)

tools = [search_tool]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant."
)

result = agent.invoke({"messages": [{"role": "user", "content": "Give me a news about Lahore weather today"}]})
print(result)