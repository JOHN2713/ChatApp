import os
from dotenv import load_dotenv
import streamlit as st
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import langchain

langchain.verbose = False
load_dotenv()

def process_html(html_file):
    html_text = html_file.read()
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text()
    return text

def process_xml(xml_files):
    xml_texts = []
    for xml_file in xml_files:
        try:
            xml_text = xml_file.read().decode('utf-8', errors='ignore')  # Intenta decodificar el contenido como utf-8 y ignora errores
        except UnicodeDecodeError:
            # Si la decodificación falla, intenta decodificar como latin1
            xml_text = xml_file.read().decode('latin1', errors='ignore')
        soup = BeautifulSoup(xml_text, "lxml")
        text = soup.get_text()
        xml_texts.append(text)
    combined_text = ' '.join(xml_texts)
    return combined_text

def main():
    st.title("QIA-Preguntas archivos html y xml")

    html_file = st.file_uploader("Sube tu archivo HTML", type="html", key="html_file")
    xml_files = st.file_uploader("Sube tus archivos XML", type="xml", accept_multiple_files=True, key="xml_files")

    combined_text = ""

    if html_file is not None:
        html_text = process_html(html_file)
        combined_text += html_text + " "

    if xml_files is not None:
        xml_text = process_xml(xml_files)
        combined_text += xml_text

    if combined_text:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(combined_text)

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        knowledge_base = FAISS.from_texts(chunks, embeddings)

        query = st.text_input('Escribe tu pregunta...')
        cancel_button = st.button('Cancelar')

        if cancel_button:
            st.stop()

        if query:
            docs = knowledge_base.similarity_search(query)
            model = "gpt-3.5-turbo-instruct"
            temperature = 0
            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cost:
                response = chain.invoke(input={"question": query, "input_documents": docs})
                print(cost)
                st.write(response["output_text"])

if __name__ == "__main__":
    main()
