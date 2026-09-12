import os
import requests
import streamlit as st
import certifi

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_tavily import TavilySearch

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.agents import (
    create_tool_calling_agent,
    AgentExecutor
)


# ==========================================
# LOAD ENV VARIABLES
# ==========================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not WEATHERSTACK_API_KEY:
    raise ValueError("WEATHERSTACK_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")


# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")

st.markdown("Search + Weather AI Agent using LangChain")


# ==========================================
# SEARCH TOOL
# ==========================================

search_tool = TavilySearch(
    max_results=2
)


# ==========================================
# WEATHER TOOL
# ==========================================

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


# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY
)


# ==========================================
# PROMPT
# ==========================================

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

    MessagesPlaceholder(
        variable_name="agent_scratchpad"
    )
])


# ==========================================
# TOOLS
# ==========================================

tools = [
    search_tool,
    get_weather_data
]


# ==========================================
# CREATE AGENT
# ==========================================

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# ==========================================
# EXECUTOR
# ==========================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


# ==========================================
# UI INPUT
# ==========================================

user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the latest news about Nepal and current weather in Kathmandu"
)


# ==========================================
# RUN AGENT
# ==========================================

if st.button("Run Agent"):

    if user_query:

        with st.spinner("Agent is thinking..."):

            try:

                response = agent_executor.invoke({
                    "input": user_query
                })

                st.success("Response Generated")

                st.markdown("## Final Response")

                st.write(response["output"])

            except Exception as e:

                st.error(f"Error: {str(e)}")

    else:

        st.warning("Please enter a query")