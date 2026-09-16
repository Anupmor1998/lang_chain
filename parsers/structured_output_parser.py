from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

# this code shows the flow without using output parser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it",task="text-generation", provider="featherless-ai")

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic')
]
parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template="List 3 facts about {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.invoke({"topic": "Python programming language"})

# result = model.invoke(prompt)

# final_Result = parser.parse(result.content)

chain = template | model | parser

result = chain.invoke({"topic": "Python programming language"})



print(result)