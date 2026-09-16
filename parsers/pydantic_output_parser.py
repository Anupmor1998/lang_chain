from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# this code shows the flow without using output parser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it",task="text-generation", provider="featherless-ai")

model = ChatHuggingFace(llm=llm)

# Schema

class Person(BaseModel):

    name: str = Field(description='name of the person')
    age: int = Field(gt=18,description='age of the person')
    city: str = Field(description='name of the city person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template="Give me name, age and city of the {place} person. \n {format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"place": "India"})

print(result)