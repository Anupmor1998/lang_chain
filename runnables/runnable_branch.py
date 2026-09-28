from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI();

parser = StrOutputParser();

prompt1 = PromptTemplate(template="Write a detailed report on the {topic}.",input_variables=['topic'])

prompt2 = PromptTemplate(template="Summarize the following \n {text}.",input_variables=['text'])


report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

result = final_chain.invoke({'topic':'Abijeet Dipke Cocroach Janta Party Head'})

print("Result: ",result)