import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system", "You are a Linkedin post generator. You will be given a topic and you will generate a LinkedIn post about that topic. The post should be engaging, informative, and professional. It should be between 100-200 words. Do not include any hashtags or emojis in the post."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a viral LinkedIn content expert. Review the LinkedIn post below "
        "and provide detailed critique and concrete recommendations to improve it "
        "(tone, structure, hook, clarity, engagement)."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_output_tokens=700)
llm = ChatOpenAI(
    model="openai/gpt-5.6-luna-pro",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=500 
)
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm