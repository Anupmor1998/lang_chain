from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI();

prompt1 = PromptTemplate(template='Generate a detailed report about {topic}.',input_variables=['topic'])

prompt2 = PromptTemplate(template='Generate a 5 point summary of the following\n {text}.',input_variables=['text'])

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'BRICS Summit 2026'})

print(result)

chain.get_graph().print_ascii()