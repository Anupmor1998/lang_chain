from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv();

model = ChatOpenAI(model="gpt-4")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who won the world series in 2020?"),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)