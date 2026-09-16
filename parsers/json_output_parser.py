from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# this code shows the flow without using output parser

load_dotenv()

model = ChatOpenAI()

parser = JsonOutputParser()

template = PromptTemplate(template="Give me the name, age and city of a fictional person in India\n {format_instructions}",input_variables=[],partial_variables={'format_instructions':parser.get_format_instructions()})


# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# we can also do the above steps using chain also 

chain = template | model | parser

final_result = chain.invoke({})

print(final_result)
print(final_result['name'])
print(type(final_result))