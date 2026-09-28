from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI();

parser = StrOutputParser();

prompt = PromptTemplate(template="Write a joke on {topic}.",input_variables=['topic'])


joke_generator_chain = RunnableSequence(prompt, model, parser)

def word_counter(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(), # here pass through will return the output same as input provided it will not run any llm or parsing just return the input as output
    'words':RunnableLambda(word_counter) # here it will convert any python login into runnable so that we can use it with the chains
})

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

result = final_chain.invoke({'topic':'Abijeet Dipke Cocroach Janta Party Head'})

print("Joke: ",result['joke'],"\n")
print("Words: ",result['words'])