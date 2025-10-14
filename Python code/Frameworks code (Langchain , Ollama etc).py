#LANGGRAPH example

from llanggraph import create_react_agent 


def get_weather(city:str) -> str:

    return f"The weather in {city} is sunny with a high of 75°F."

agent = create_react_agent(
    model = "anthropic:claude-3-7-sonnet-latest",
    tools = [get_weather],
    prompt = 'You are helpful assistant'q
)

#run the agent

agent.invoke("What's the weather like in New York?")


#Langchain example

import getpass
import os 

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    from langchain.chat_models import ChatOpenAI

    model = ChatOpenAI("gpt-4o", model_provider = 'google_genai')


model.invoke('Hello, World! ')