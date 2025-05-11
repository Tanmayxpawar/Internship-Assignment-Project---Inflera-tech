import streamlit as st
from knowledge_assistant import KnowledgeAssistant
import os
from dotenv import load_dotenv
import time
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the knowledge assistant
@st.cache_resource
def get_assistant():
    return KnowledgeAssistant()

def create_chain(assistant):
    """Create a robust chain with memory and context"""
    prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable assistant with access to various tools and document information. Engage naturally and answer like ChatGPT—friendly, clear, and conversational.

Chat History:
{chat_history}

Document Context:
{document_context}

Current Query:
{question}

Instructions:
1. If the query is a calculation request, use the calculator tool and provide the result clearly.
2. If the query is about word definitions, provide a clear and concise definition.
3. If the query is about document content, use the provided context to give a detailed answer.
4. If no specific tool matches and no context is available, provide a helpful response based on your knowledge.
5. Maintain a natural, conversational flow.
6. If asked generic questions, keep answers concise.
7. If user greets you, greet them back warmly.
8. If asked about previous questions or conversation history, use the chat history to provide a summary.

Response:""")

    def get_chat_history(inputs):
        # Get the full chat history from memory
        memory_variables = assistant.memory.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", [])
        if not chat_history:
            return "No previous conversation history."
        return "\n".join([
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) 
            else f"Assistant: {msg.content}"
            for msg in chat_history
        ])

    def format_document_context(result):
        if not result.get('context'):
            return "No relevant document context available."
        return result['context']

    # Create the chain with memory
    chain = (
        {
            "document_context": lambda x: format_document_context(assistant.process_query(x)),
            "question": RunnablePassthrough(),
            "chat_history": get_chat_history
        }
        | prompt
        | assistant.llm
        | StrOutputParser()
    )
    
    return chain

def main():
    # Create sidebar for file upload
    with st.sidebar:
        st.title("🤖 Knowledge Assistant")
        
        # File upload section
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
        
        # Process button
        process_button = st.button("Process Documents", type="primary")
        
        # Status section
        st.subheader("📊 Status")
        status_placeholder = st.empty()
        
        # Example queries
        st.subheader("💡 Example Queries")
        st.markdown("""
        - What is the capital of France?
        - Calculate 500 * 200
        - Define the word 'serendipity'
        - What are the key features of the product?
        - What questions have I asked you so far?
        """)

    # Main chat area
    st.title("Chat")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Create a container for chat messages with fixed height and scrolling
    chat_container = st.container()
    
    # Display chat history in reverse order (newest at bottom)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "tool_used" in message:
                    st.caption(f"Tool used: {message['tool_used']}")
    
    # Chat input at the bottom
    user_input = st.chat_input("Enter your question...", key="chat_input")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Update status
        status_placeholder.info("Processing your request...")
        
        # Initialize assistant and chain if not already done
        if 'assistant' not in st.session_state:
            st.session_state.assistant = get_assistant()
            st.session_state.chain = create_chain(st.session_state.assistant)
        
        # Process query using the chain
        with st.spinner("Thinking..."):
            # Get tool information
            tool_info = st.session_state.assistant.process_query(user_input)
            # Get response from chain
            response = st.session_state.chain.invoke(user_input)
            # Save the response to memory
            st.session_state.assistant.memory.save_context(
                {"input": user_input},
                {"output": response}
            )
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "tool_used": tool_info["tool_used"]
        })
        
        with st.chat_message("assistant"):
            st.write(response)
            st.caption(f"Tool used: {tool_info['tool_used']}")
        
        # Update status
        status_placeholder.success("Request completed!")
    
    # Handle file processing when the process button is clicked
    if process_button and uploaded_files:
        status_placeholder.info("Processing documents...")
        try:
            temp_dir = "temp_docs"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save and process uploaded files
            for file in uploaded_files:
                with open(os.path.join(temp_dir, file.name), "wb") as f:
                    f.write(file.getvalue())
            
            # Process documents
            if 'assistant' not in st.session_state:
                st.session_state.assistant = get_assistant()
            st.session_state.assistant.ingest_documents(temp_dir)
            status_placeholder.success("Documents processed successfully!")
        except Exception as e:
            status_placeholder.error(f"Error processing documents: {str(e)}")

if __name__ == "__main__":
    main() 