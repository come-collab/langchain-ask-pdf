#Recreating ask pdf with huggingface models

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain



def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your documents")
    st.header("Ask your document 💬")
    
    #upload file 
    pdf = st.file_uploader("Upload our PDF",type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
              # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-base-en-v1.5")
      knowledge_base = FAISS.from_texts(chunks,embeddings)
    
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs ,question=user_question)
        st.write(response)
if __name__ == '__main__':
    main()
