import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool


os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

if not WEATHERSTACK_API_KEY:
    raise ValueError("WEATHERSTACK_API_KEY is missing from .env")


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY
)


search_tool = TavilySearch(
    max_results=2
)


@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


tools = [
    search_tool,
    get_weather_data
]


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        You have access to multiple tools.

        Use the available tools when the user's question requires
        current information or external information.

        IMPORTANT:
        - Complete all parts of the user's request.
        - If multiple tools are needed, use all necessary tools.
        - If you use multiple tools, use the results from ALL tools
          in your final answer.
        - Do not ignore the result of any tool you called.
        - Clearly separate different pieces of information when useful.
        - Give clear and accurate answers.
        """
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


response = agent_executor.invoke({
    "input": "Find the latest news about the Rasuwa Nepal flood, how many people lost their lives and then tell me the current weather in Kathmandu."
})


print("\n================ FINAL ANSWER ================\n")
print(response["output"])