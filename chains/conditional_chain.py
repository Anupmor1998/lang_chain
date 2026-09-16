from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI();

class Feedback(BaseModel):

    sentiment: Literal['positive','negative'] = Field(description='sentiment of the feedback')

parser1 = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(template='Classify the sentiment of the following feedback into positive or negative  \n {feedback} \n {format_instructions}.',input_variables=['feedback'],partial_variables={'format_instructions':parser2.get_format_instructions()})

prompt2 = PromptTemplate(template='Provide an appropriate response of the following positive  \n {feedback}.',input_variables=['feedback'])

prompt3 = PromptTemplate(template='Provide an appropriate response of the following negative  \n {feedback}.',input_variables=['feedback'])


classifier_chain = prompt1 | model | parser2


branched_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x:'Could not find any appropriate sentiment')
)

chain = classifier_chain | branched_chain

result = chain.invoke({'feedback':'this is a wonderful phone'})

print(result)

chain.get_graph().print_ascii()