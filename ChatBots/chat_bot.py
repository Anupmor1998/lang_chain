from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv();

model = ChatOpenAI(model="gpt-4")

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if(user_input.lower() == "exit"):
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Bot: ", result.content)

print("Chat History: ", chat_history)