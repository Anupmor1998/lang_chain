from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI();

parser = StrOutputParser();

prompt1 = PromptTemplate(template="Write a joke on {topic}.",input_variables=['topic'])

prompt2 = PromptTemplate(template="Explain the following joke - {text}.",input_variables=['text'])

joke_generator_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(), # here pass through will return the output same as input provided it will not run any llm or parsing just return the input as output
    'explain':RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

result = final_chain.invoke({'topic':'Abijeet Dipke Cocroach Janta Party Head'})

print("Joke: ",result['joke'],"\n\n")
print("Explanation: ",result['explain'])