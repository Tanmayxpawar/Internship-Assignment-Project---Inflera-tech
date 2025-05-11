# Inflera Technologies - Knowledge Assistant Chatbot
Internship Assignment Project

## Overview
This is a knowledge assistant chatbot built for Inflera Technologies Private Limited as part of an internship assignment. The system is designed to process and query PDF documents, with a focus on FAQ documents.

## Live Demo
Access the live application at: 

## Pre-processed Data
- The system comes with 3 pre-processed FAQ PDFs
- The processed data is stored in the `chroma_db` directory
- No need to re-process these documents to start querying

## Features
1. **Document Processing**
   - Upload and process new PDF documents
   - Automatic text extraction and chunking
   - Vector embeddings generation for semantic search

2. **Query Capabilities**
   - Natural language queries on processed documents
   - Mathematical calculations
   - Word definitions
   - Context-aware responses

3. **User Interface**
   - Clean Streamlit-based web interface
   - Real-time document processing
   - Interactive chat experience
   - Tool usage transparency

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your environment:
   - Create a `.env` file in the project root
   - Add your GROQ API key: `GROQ_API_KEY=your_api_key_here`
   - (Optional) Configure model settings if needed
4. Run the application: `streamlit run app.py`

## Deployment
The application is deployed on Streamlit Cloud:
1. Fork this repository
2. Sign up for a free Streamlit Cloud account
3. Connect your forked repository
4. Add your GROQ API key in the Streamlit Cloud secrets management
5. Deploy the application

### Deployment Requirements
- Python 3.9 or higher
- All dependencies are specified in `requirements.txt`
- Make sure to use the exact versions specified to avoid compatibility issues
- The application requires about 1GB of RAM to run smoothly

### Streamlit Cloud Configuration
1. Set Python version to 3.9
2. Add the following to your Streamlit Cloud secrets:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
3. Set the main file as `app.py`

## Important Note About API Keys
For security reasons, the `.env` file is not included in the repository. You will need to:
1. Sign up for a GROQ API key at https://console.groq.com
2. Create your own `.env` file with your API key
3. Never share your API key or commit it to version control

## Usage
1. The system is ready to query pre-processed FAQ documents
2. To add new documents:
   - Use the sidebar upload feature
   - Click "Process Documents"
   - Wait for processing confirmation
3. Start chatting with the assistant

## Technical Implementation
- Built with Python and Streamlit
- Uses LangChain for LLM integration
- ChromaDB for vector storage
- Sentence Transformers for embeddings

## Note
This implementation meets the basic requirements specified in the assignment. The system can be enhanced with:
- Cloud-based vector storage
- User authentication
- Advanced document processing
- Custom model fine-tuning
- Additional query capabilities

## Contact
For any queries or improvements, please contact @ pawartanmay95@gmail.com. 
