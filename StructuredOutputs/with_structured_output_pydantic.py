from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional,Literal
from pydantic import BaseModel,Field

load_dotenv()

model = ChatOpenAI()

# Schema

class Review(BaseModel): #this class will return the pydantic object so to use the dictionary appraoch to to access value we have convert it like dict(review)
   
    summary: str = Field(description="A brief summary of the review in 1 or 2 sentences")
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    sentiment:Literal['pos', 'neg', 'neut'] = Field(description="The overall sentiment of the review, can be positive, negative or neutral")
    pros: Optional[list[str]] = Field(default=None,description="A list of pros mentioned in the review, if any")
    cons: Optional[list[str]] = Field(default=None,description="A list of cons mentioned in the review, if any")
    name: Optional[str] = Field(default=None,description="The name of the reviewer, if they chose to include it")

# Review 

prompt = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Anup Mor
"""

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(prompt)
# result will be of type pydantic object and we can access the values using the attributes of the object
print(result.name)
# converting the pydantic object to dictionary to access the values using the keys of the dictionary
result_dict = dict(result)

print(result_dict['summary'])
print(result_dict['sentiment'])