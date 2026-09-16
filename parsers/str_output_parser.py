from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# this code shows the flow without using output parser

load_dotenv()

model = ChatOpenAI()

# prompt 1

template1 = PromptTemplate(template="Please generate a detailed report on {topic}.",input_variables=['topic'])

# prompt 2 

template2 = PromptTemplate(template="Please write a 5 line summary on \n {text}",input_variables=['text'])

prompt1 = template1.invoke({'topic':'Men Mental Health'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)