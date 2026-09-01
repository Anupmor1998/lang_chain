from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv();

model = ChatOpenAI(model="gpt-4")

st.header("Research Tool")

# User Inputs
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# template 

template = load_prompt("template.json")


# fill the placeholders in the template with user inputs
# prompt = template.invoke({
#     "paper_input": paper_input,
#     "style_input": style_input,
#     "length_input": length_input
# })


if st.button("Summarize"):
    #using chain to invoke the model with the filled prompt
    chain = template | model
    result = chain.invoke({"paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input})
    st.write(result.content)