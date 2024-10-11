import time
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to process PDF and create vector store
def process_pdf(pdf_file):
    # Extract text from PDF
    raw_text = extract_text_from_pdf(pdf_file)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    texts = text_splitter.split_text(raw_text)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    return vectorstore

def get_relevant_chunks(query, vectorstore, k=5):
    return vectorstore.similarity_search(query, k=k)

def format_context(relevant_chunks):
    return "\n\n".join([chunk.page_content for chunk in relevant_chunks])

def setup_qa_system(pdf_file):
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.5,
        max_retries=3,
        top_p=0.95,
        max_tokens=3000,
        presence_penalty=0.1,
        frequency_penalty=0.1,
        safe_mode=False,
        random_seed=42,
    )

    # Process PDF and create vector store
    vectorstore = process_pdf(pdf_file)

    # Create ConversationSummaryBufferMemory
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an intelligent assistant specializing in answering questions about a specific PDF document while also engaging in general conversation. Your role is to:

            1. Distinguish between questions about the PDF content and general conversation or greetings.
            2. For PDF-related questions:
               a. Carefully analyze the given context and chat history.
               b. Provide accurate, relevant, and comprehensive answers derived from the context and chat history.
               c. If the context or chat history contains multiple relevant pieces of information, synthesize them into a coherent response.
               d. If the question cannot be fully answered based on the given information, clearly state what is missing or unclear.
               e. If the answer cannot be found in the context or chat history at all, respond with "I don't have enough information from the PDF to answer that question."
            3. For general conversation or greetings:
               a. Respond naturally and appropriately without referencing the PDF content.
               b. Maintain a friendly and professional tone.
            4. Always maintain a helpful and professional demeanor.
            5. If appropriate, suggest follow-up questions that could provide more clarity or depth on the topic.

            Remember, for PDF-related questions, your knowledge is limited to the provided context and chat history. Do not introduce external information or make assumptions beyond what is explicitly stated in the PDF content."""
        ),
        (
            "human",
            """Context:
            {context}
            Chat History:
            {chat_history}
            User Input: {question}
            Please provide an appropriate response based on whether this is a PDF-related question or general conversation."""
        ),
    ])

    # Create the chain
    chain = prompt | llm

    return vectorstore, memory, chain

def ask_question(question, vectorstore, memory, chain):
    relevant_chunks = get_relevant_chunks(question, vectorstore)
    context = format_context(relevant_chunks)
    
    # Get chat history from memory
    chat_history = memory.load_memory_variables({})["history"]
    
    # Invoke the chain
    response = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })
    
    # Save the interaction to memory
    memory.save_context({"input": question}, {"output": response.content})
    
    return response.content.strip()

def main():
    load_dotenv()
    
    st.set_page_config(page_title="PDF Q&A Chatbot", page_icon=":book:")
    st.title("PDF Q&A Chatbot 📚")

    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

    if uploaded_file is not None:
        # Setup QA system
        vectorstore, memory, chain = setup_qa_system(uploaded_file)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            welcome_message = (
                "Hello! I'm your PDF Q&A assistant. I'm here to help you with questions about the content of the uploaded PDF. "
                "What would you like to know?"
            )
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        if prompt := st.chat_input("Ask a question about the PDF:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                response_text = ask_question(prompt, vectorstore, memory, chain)
                placeholder = st.empty()
                accumulated_response = ""
                for char in response_text:
                    accumulated_response += char
                    placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                    time.sleep(0.01)

            st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        st.write("Please upload a PDF file to start the Q&A session.")

if __name__ == "__main__":
    main()