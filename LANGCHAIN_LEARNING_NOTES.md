# 📓 My Personal LangChain Handbook & Revision Notes

> _"Write it down like you're explaining it to your future self over coffee — no corporate jargon, just intuition, practical code, and why things actually work."_

---

## 🗺️ Quick Roadmap of Everything in This Repo

| Section                         | Folder in Repo         | Core Question Answered                                                    |
| :------------------------------ | :--------------------- | :------------------------------------------------------------------------ |
| **1. The Core Engines**         | `LLMs/`, `ChatModels/` | _What is the difference between an old LLM and a modern Chat Model?_      |
| **2. Dynamic Prompting**        | `Prompts/`             | _Why can't I just use Python f-strings? How do templates save & load?_    |
| **3. Roles & Memory**           | `ChatBots/`            | _How do System, Human, and AI messages work with chat history?_           |
| **4. Output Parsers**           | `parsers/`             | _How do I stop getting raw text and get clean strings, JSON, or objects?_ |
| **5. Structured Outputs**       | `StructuredOutputs/`   | _What makes `with_structured_output()` different from old parsers?_       |
| **6. LCEL & Chains**            | `chains/`              | _How does the pipe `\|` operator connect prompt, model, and parser?_      |
| **7. Runnables Under the Hood** | `runnables/`           | _What are Passthrough, Lambda, Parallel, and Branch doing internally?_    |
| **8. Embeddings & Search**      | `EmbeddingModels/`     | _How does text turn into vectors and how does similarity search work?_    |

---

## 1. The Core Engines: LLMs vs. ChatModels

### 🧠 The Intuition

- **Old LLMs (`OpenAI`)**: Think of it as an **autocomplete machine**. You give it a piece of text (string in), it guesses what text comes next (string out). It doesn't know about "users" or "assistants".
- **Chat Models (`ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatHuggingFace`)**: Built on top of modern instruction-tuned models. They don't just complete text; they participate in a **dialogue**. They accept structured lists of _Messages_ and return an `AIMessage` containing text, token metadata, and tool calls.

```
Old LLM:       "The capital of France is"  ───> "Paris."
Chat Model:    [SystemMessage, HumanMessage] ───> AIMessage(content="Paris is...")
```

### 💻 Code Patterns from Our Repo

#### Old-School LLM ([LLMs/1_llm_demo.py](file:///home/dreamworld/Desktop/Langchain/LLMs/1_llm_demo.py))

```python
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(model="gpt-3.5-turbo-instruct")
result = llm.invoke("What is the Capital of India?")
print(result) # Raw string output directly
```

#### Modern Chat Models ([ChatModels/](file:///home/dreamworld/Desktop/Langchain/ChatModels))

```python
# OpenAI
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")
res = model.invoke("What is the Capital of India?")
print(res.content) # Notice: res is an AIMessage object, text is in .content!

# Google Gemini (ChatModels/2_chatmodel_google.py)
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# HuggingFace Endpoint (ChatModels/3_chatmodel_hf_api.py)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", provider="featherless-ai")
model = ChatHuggingFace(llm=llm)
```

> 💡 **Golden Rule**: Always use `ChatModels` for modern apps. Old `LLMs` are legacy and don't support tool calling, structured outputs, or multi-turn conversational context cleanly.

---

## 2. Dynamic Prompts & Templates

### 🧠 The Intuition

Why not just use Python's `f"Summarize {paper}"`?

1. **Reusability & Serialization**: LangChain `PromptTemplate` can be saved to disk (`template.save("template.json")`) and loaded in production apps (`load_prompt("template.json")`) without hardcoding prompt strings into your UI code.
2. **Validation**: It checks that all declared variables (`input_variables=['topic']`) match the placeholders, catching missing values before calling expensive APIs.
3. **Composable Chains**: A `PromptTemplate` is a `Runnable`. It slots cleanly into the pipe `|` operator!

### 💻 Key Lessons from Our Code

#### 1. Generating & Saving Prompts ([Prompts/prompt_generator.py](file:///home/dreamworld/Desktop/Langchain/Prompts/prompt_generator.py))

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""Please summarize the research paper titled "{paper_input}"...
    Style: {style_input}
    Length: {length_input}""",
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True
)

template.save("template.json") # Persist to JSON for separation of concerns!
```

#### 2. Loading into a Streamlit UI ([Prompts/prompt_template_ui.py](file:///home/dreamworld/Desktop/Langchain/Prompts/prompt_template_ui.py))

```python
from langchain_core.prompts import load_prompt

template = load_prompt("template.json")
chain = template | model # Pipe template directly to model!

# When user clicks button:
result = chain.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})
st.write(result.content)
```

---

## 3. ChatBots: Messages, Roles & Memory

### 🧠 The Intuition: The 3 Musketeers of Chat

Chat models don't have natural memory. Every HTTP request is stateless. To give a bot "memory", you must resend the conversation history every time. LangChain represents this using 3 core message types:

1. `SystemMessage`: The hidden rules of the universe for the bot ("You are an angry pirate", "You are a customer support agent").
2. `HumanMessage`: What the user actually typed.
3. `AIMessage`: What the model replied with.

### 💻 Code Patterns from Our Repo

#### The Basic Conversational Loop ([ChatBots/chat_bot.py](file:///home/dreamworld/Desktop/Langchain/ChatBots/chat_bot.py))

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # 1. Append user's message
    chat_history.append(HumanMessage(content=user_input))

    # 2. Invoke with FULL history
    result = model.invoke(chat_history)

    # 3. Store AI's reply back to history for next turn!
    chat_history.append(AIMessage(content=result.content))
    print("Bot:", result.content)
```

#### Dynamic Chat Prompts & MessagesPlaceholder ([ChatBots/message_placeholder.py](file:///home/dreamworld/Desktop/Langchain/ChatBots/message_placeholder.py))

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support assistant.'),
    MessagesPlaceholder(variable_name='chat_history'), # Injects a list of messages right here!
    ('user', '{query}')
])

prompt = chat_template.invoke({
    'chat_history': [HumanMessage(content="Hi"), AIMessage(content="Hello!")],
    'query': 'Where is my order?'
})
```

> 💡 **Why `MessagesPlaceholder`?** It allows you to dynamically slot a varying-length list of prior messages right into the middle of a structured prompt template without messy string formatting.

---

## 4. Output Parsers: Taming the Model's Output

### 🧠 The Intuition

When a model responds, it gives you an `AIMessage` or a chatty string with commentary ("Sure! Here is the JSON: ...").
If you are building code, you don't want conversational fluff. You want:

- A clean string (`StrOutputParser`)
- A Python dictionary (`JsonOutputParser`, `StructuredOutputParser`)
- A validated Python class instance (`PydanticOutputParser`)

### 🛠️ Comparison Table of Parsers

| Parser                       | Returns         | Best Used When                                                     | Found in File                                                                                                        |
| :--------------------------- | :-------------- | :----------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **`StrOutputParser`**        | Pure `str`      | Connecting model output to another prompt in LCEL                  | [parsers/str_output_parser.py](file:///home/dreamworld/Desktop/Langchain/parsers/str_output_parser.py)               |
| **`JsonOutputParser`**       | Python `dict`   | Need JSON without strict field validation                          | [parsers/json_output_parser.py](file:///home/dreamworld/Desktop/Langchain/parsers/json_output_parser.py)             |
| **`StructuredOutputParser`** | Python `dict`   | Legacy multi-field extraction with schemas                         | [parsers/structured_output_parser.py](file:///home/dreamworld/Desktop/Langchain/parsers/structured_output_parser.py) |
| **`PydanticOutputParser`**   | Pydantic Object | Need strict type enforcement + custom validation (e.g. `age > 18`) | [parsers/pydantic_output_parser.py](file:///home/dreamworld/Desktop/Langchain/parsers/pydantic_output_parser.py)     |

### 💻 The Magic Trick: `partial_variables` & Format Instructions

Notice how our parser code injects formatting instructions automatically:

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class Person(BaseModel):
    name: str = Field(description='name of the person')
    age: int = Field(gt=18, description='age of the person')
    city: str = Field(description='city')

parser = PydanticOutputParser(pydantic_object=Person)

# parser.get_format_instructions() returns the exact JSON schema instructions LLM needs!
template = PromptTemplate(
    template="Give me person details for {place}.\n{format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"place": "India"})
# result is an instance of Person! (result.name, result.age)
```

---

## 5. Modern Structured Outputs: `with_structured_output()`

### 🧠 The Giant Leap: Parsers vs. `with_structured_output()`

- **Old Output Parsers**: Beg the model in the prompt: _"Please format your response as JSON matching this schema: ..."_ If the model hallucinates a prefix like _"Sure, here it is:"_, the parser crashes!
- **`with_structured_output()`**: Uses native **Function Calling / Tool Calling** at the API level (supported by OpenAI, Gemini, Claude). The model is forced by the model engine to output valid JSON matching your schema.

```
Prompt -> Model -> Raw text -> Regex/Parser -> Python Object  ❌ (Fragile, can break)
Prompt -> Model(with_structured_output) ------> Python Object  ✅ (Guaranteed schema)
```

### 💻 3 Ways We Implemented It in `StructuredOutputs/`:

#### Option A: Pydantic Schema ([StructuredOutputs/with_structured_output_pydantic.py](file:///home/dreamworld/Desktop/Langchain/StructuredOutputs/with_structured_output_pydantic.py))

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal

class Review(BaseModel):
    summary: str = Field(description="1-2 sentence summary")
    sentiment: Literal['pos', 'neg', 'neut'] = Field(description="Overall sentiment")
    pros: Optional[list[str]] = Field(default=None)

structured_model = model.with_structured_output(Review)
res = structured_model.invoke(prompt_text)
# res is a Review instance! res.summary, res.sentiment
```

#### Option B: TypedDict + Annotated ([StructuredOutputs/with_structured_output_typed_dict.py](file:///home/dreamworld/Desktop/Langchain/StructuredOutputs/with_structured_output_typed_dict.py))

```python
from typing import TypedDict, Annotated, Literal

class Review(TypedDict):
    # Annotated lets you attach docstrings to fields so the LLM knows what to put!
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal['pos', 'neg', 'neut'], "Sentiment tag"]

structured_model = model.with_structured_output(Review)
res = structured_model.invoke(prompt_text)
# res is a pure Python dictionary! res['summary']
```

#### Option C: Raw JSON Schema ([StructuredOutputs/with_structured_output_json_schema.py](file:///home/dreamworld/Desktop/Langchain/StructuredOutputs/with_structured_output_json_schema.py))

```python
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["pos", "neg"]}
    },
    "required": ["summary", "sentiment"]
}
structured_model = model.with_structured_output(json_schema)
```

> 🎯 **Which one should I choose in real projects?**
>
> - **Use Pydantic** when you need data validation (e.g., regex, min values, defaults).
> - **Use TypedDict** when you just want a standard Python `dict` without heavy Pydantic dependencies.
> - **Use Output Parsers** ONLY when using local open-source models (like TinyLlama) that don't support native tool calling!

---

## 6. LCEL (LangChain Expression Language) & Chains

### 🧠 The Intuition: The Unix Pipe `|`

In Linux bash: `cat file.txt | grep "error" | wc -l`.
Data flows from left to right. LangChain took this exact philosophy:

```
input ──> [PromptTemplate] ──> [ChatModel] ──> [StrOutputParser] ──> output
                 prompt             AIMessage            clean str
```

Syntax:

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "Cricket"})
```

### ⛓️ The 4 Chain Patterns in Our Repo

#### 1. Simple Linear Chain ([chains/simple_chain.py](file:///home/dreamworld/Desktop/Langchain/chains/simple_chain.py))

`template | model | parser`

#### 2. Sequential Chain (Pass output of step 1 into step 2) ([chains/sequential_chain.py](file:///home/dreamworld/Desktop/Langchain/chains/sequential_chain.py))

```python
prompt1 = PromptTemplate(template='Generate a detailed report about {topic}.', input_variables=['topic'])
prompt2 = PromptTemplate(template='Generate a 5 point summary of the following:\n {text}.', input_variables=['text'])

# StrOutputParser is essential here! It strips AIMessage so prompt2 receives pure {text}!
chain = prompt1 | model | parser | prompt2 | model | parser
res = chain.invoke({'topic': 'BRICS Summit 2026'})
```

#### 3. Parallel Chain (Run tasks concurrently) ([chains/parallel_chain.py](file:///home/dreamworld/Desktop/Langchain/chains/parallel_chain.py))

```python
prompt1 = PromptTemplate(template='Generate short notes on: {text}')
prompt2 = PromptTemplate(template='Generate a 5-question quiz on: {text}')
prompt3 = PromptTemplate(template='Merge notes: {notes} and quiz: {quiz}')

# RunnableParallel takes input once, sends to both branches simultaneously!
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz':  prompt2 | model2 | parser
})

# Then pipe into the merger step!
full_chain = parallel_chain | prompt3 | model1 | parser
```

#### 4. Conditional Chain (Branching based on LLM classification) ([chains/conditional_chain.py](file:///home/dreamworld/Desktop/Langchain/chains/conditional_chain.py))

1. Step 1: LLM classifies customer sentiment into `"positive"` or `"negative"` via Pydantic.
2. Step 2: `RunnableBranch` inspects the sentiment and routes to the appropriate response chain!

```python
classifier_chain = prompt_classify | model | pydantic_parser

branched_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt_positive | model | str_parser),
    (lambda x: x.sentiment == 'negative', prompt_negative | model | str_parser),
    RunnableLambda(lambda x: 'Could not determine sentiment') # default fallback
)

chain = classifier_chain | branched_chain
```

> 🔍 **Pro-tip**: You can visualize any LCEL chain graph in the terminal using:
> `chain.get_graph().print_ascii()`

---

## 7. Runnables: The Secret Engine Under the Hood

### 🧠 The Intuition

Every component in LCEL (prompts, models, parsers) implements the **Runnable Interface**.
That's why `a | b` works. `a | b` is actually syntactic sugar for `RunnableSequence(a, b)`!

### 🧱 The 5 Core Runnables in `runnables/`

```
┌───────────────────────┬───────────────────────────────────────────────────────────────┐
│ Runnable              │ What It Does in Plain English                                │
├───────────────────────┼───────────────────────────────────────────────────────────────┤
│ RunnableSequence      │ Runs A, passes result to B, then to C (sequential pipeline)  │
│ RunnableParallel      │ Takes 1 input, runs multiple branches at once, outputs a dict │
│ RunnablePassthrough   │ Passes input forward unchanged without touching it            │
│ RunnableLambda        │ Wraps ANY standard Python function so it can live in a chain  │
│ RunnableBranch        │ If/Else condition routing for chains                          │
└───────────────────────┴───────────────────────────────────────────────────────────────┘
```

### 💻 Deep Dive into Each Runnable

#### A. `RunnableSequence` ([runnables/runnable_sequence.py](file:///home/dreamworld/Desktop/Langchain/runnables/runnable_sequence.py))

```python
# These two lines do the EXACT same thing:
chain = prompt1 | model | parser | prompt2 | model | parser
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
```

#### B. `RunnableParallel` ([runnables/runnable_parallel.py](file:///home/dreamworld/Desktop/Langchain/runnables/runnable_parallel.py))

```python
parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt_tweet, model, parser),
    'linkedin': RunnableSequence(prompt_linkedin, model, parser)
})
result = parallel_chain.invoke({'topic': 'AI Revolution'})
# result is a dict: {'tweet': '...', 'linkedin': '...'}
```

#### C. `RunnablePassthrough` ([runnables/runnable_pass_through.py](file:///home/dreamworld/Desktop/Langchain/runnables/runnable_pass_through.py))

**The Problem**: Suppose Step 1 generates a joke. In Step 2, you want to explain the joke, but in the final output you want BOTH the original joke AND the explanation!
If you just piped joke into explain, the original joke is lost!
**The Solution**:

```python
joke_chain = prompt1 | model | parser # outputs: "Why did the chicken..."

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(), # Leaves the joke untouched and forwards it!
    'explain': prompt2 | model | parser # Explains the joke
})

final_chain = RunnableSequence(joke_chain, parallel_chain)
res = final_chain.invoke({'topic': 'Programmers'})
# res = {'joke': '...', 'explain': '...'}
```

#### D. `RunnableLambda` ([runnables/runnable_lambda.py](file:///home/dreamworld/Desktop/Langchain/runnables/runnable_lambda.py))

**The Problem**: What if you have a custom Python function (like counting words, formatting dates, or querying SQL) and you want it in your LCEL chain?

```python
def count_words(text: str) -> int:
    return len(text.split())

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'words': RunnableLambda(count_words) # Turns regular python function into a Runnable!
})
```

#### E. `RunnableBranch` ([runnables/runnable_branch.py](file:///home/dreamworld/Desktop/Langchain/runnables/runnable_branch.py))

Conditional If-Else routing:

```python
branch_chain = RunnableBranch(
    # (condition_func, runnable_to_execute)
    (lambda text: len(text.split()) > 500, summarize_chain),
    RunnablePassthrough() # default fallback if condition is false
)
```

---

## 8. Embedding Models & Document Similarity (RAG Basics)

### 🧠 The Intuition: What is an Embedding?

Computers don't understand words; they understand geometry.
An **Embedding Model** transforms a piece of text into a list of floating-point numbers (e.g., `[0.024, -0.912, 0.431, ...]`) representing its semantic meaning in high-dimensional space.

- Sentences with similar meanings end up **closer** in space.
- Sentences with different meanings end up **far apart**.

```
"Virat Kohli is a batsman"  ──┐
                              ├─ Cosine Similarity ≈ 0.89 (Very close!)
"Rohit Sharma hit a century" ─┘

"The capital of France is Paris" ─ Cosine Similarity ≈ 0.12 (Far away!)
```

### 💻 Methods in Our Repo

#### `embed_query` vs `embed_documents` ([EmbeddingModels/1_embedding_openai_query.py](file:///home/dreamworld/Desktop/Langchain/EmbeddingModels/1_embedding_openai_query.py) & [2_embedding_openai_docs.py](file:///home/dreamworld/Desktop/Langchain/EmbeddingModels/2_embedding_openai_docs.py))

```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

# 1. embed_query: For ONE string (e.g. user search query). Returns 1D list: [float, ...]
query_vector = embedding.embed_query("Delhi is the capital of India.")

# 2. embed_documents: For MULTIPLE strings (corpus). Returns 2D list: [[float, ...], [float, ...]]
doc_vectors = embedding.embed_documents([
    "Delhi is the capital of India.",
    "The Taj Mahal is located in Agra."
])
```

> ⚠️ **Why the distinction?** Some models (like Cohere or e5) actually embed queries differently from documents (adding prefixes like `search_query:` vs `search_document:`) to improve retrieval accuracy.

#### Running Embeddings 100% Free Locally ([EmbeddingModels/3_embedding_hf_local.py](file:///home/dreamworld/Desktop/Langchain/EmbeddingModels/3_embedding_hf_local.py))

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Runs on your local machine using sentence-transformers! Zero API fees!
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
result = embedding.embed_documents(["Delhi is the capital of India."])
```

#### Finding Most Relevant Document with Cosine Similarity ([EmbeddingModels/4_document_similarity.py](file:///home/dreamworld/Desktop/Langchain/EmbeddingModels/4_document_similarity.py))

```python
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Virat Kohli is an Indian cricketer...",
    "MS Dhoni is a former Indian captain...",
    "Rohit Sharma is known for his elegant batting...",
]
query = "Tell me about rohit sharma"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

# Compute cosine similarity between query and all documents
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the highest scoring document!
best_index, best_score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("Best match:", documents[best_index])
print("Similarity Score:", best_score)
```

> 🏆 **Congratulations**: This 10-line script is the exact mathematical foundation behind **Retrieval-Augmented Generation (RAG)**!

---

## ⚡ Super-Quick Revision Cheat Sheet (When You Have 60 Seconds)

1. **`llm.invoke(str)` vs `chat_model.invoke([messages])`**: LLM = raw text completion; ChatModel = conversation with role messages and metadata.
2. **`template | model | parser`**: LCEL pipe. Output of left becomes input of right.
3. **`StrOutputParser()`**: Always put this between two chained models so the second model gets a clean string instead of an `AIMessage`.
4. **`with_structured_output(Schema)`**: Modern way to get typed JSON/Pydantic from models via function calling.
5. **`RunnableParallel({'a': chain1, 'b': chain2})`**: Runs tasks at the same time and packages results into a dict.
6. **`RunnablePassthrough()`**: "Don't touch my data, just hand it down the line so I don't lose it."
7. **`RunnableLambda(my_func)`**: Lets you drop any regular Python function into an LCEL chain.
8. **`RunnableBranch((condition, chain), fallback)`**: If-else statement for chains.
9. **`embed_query(text)`**: 1 string -> 1 vector. For user search bar.
10. **`embed_documents([t1, t2])`**: List of strings -> List of vectors. For your knowledge base.
