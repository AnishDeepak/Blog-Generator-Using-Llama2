import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


def get_llma_response(input_text,nof_words,blog_style):
    llm=CTransformers(model='model/llama-2-7b-chat.ggmlv3.q2_K.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    template="""
    write a blog for {blog_style} job profile topic {input_text}
    within {nof_words} words.
    """
    prompt=PromptTemplate(
        input_variables=['blog_style','input_text','nof_words'],
        template=template
    )

    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,nof_words=nof_words))
    print(response)
    return response
st.set_page_config(page_title='Blog Generator',
                   page_icon='🧑‍🏫',
                   layout='centered',
                   initial_sidebar_state='collapsed'
)

st.header("Generate Blogs using LLMA-2 🧑‍🏫")
input_text=st.text_input('Enter the topic to create blog')

col1,col2=st.columns([5,5])
with col1:
    nof_words=st.text_input('Nof words')
with col2:
    blog_style=st.selectbox('write the blog for',
                            ('Research','Data Scientist','Common People'),
                            index=0)
submit=st.button('Generate')

if submit:
    st.write(get_llma_response(input_text,nof_words,blog_style))