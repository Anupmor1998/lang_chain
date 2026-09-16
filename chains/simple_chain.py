from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI();

template = PromptTemplate(template='Generate 5 interesting fasts about {topic}.',input_variables=['topic'])

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic':'Cricket'})

print(result)

# To visualize the chain we can use these methods 

chain.get_graph().print_ascii()