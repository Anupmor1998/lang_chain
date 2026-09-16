from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# this code shows the flow without using output parser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",task="text-generation", provider="featherless-ai")

model = ChatHuggingFace(llm=llm)



# prompt 1

template1 = PromptTemplate(template="Please generate a detailed report on {topic}.",input_variables=['topic'])

# prompt 2 

template2 = PromptTemplate(template="Please write a 5 line summary on the following: \n {text}",input_variables=['text'])

# Here open source model don't support inbuilt with_structured_output option like openAI, so here we have use the output parser for that and we can also use output parser for openAi models as well

# if we chaining then output parser are more handy because here we don't have to access the text using result.content

parser = StrOutputParser();

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'Men Mental Health'})

print(result)