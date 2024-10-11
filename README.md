# PDF Q&A Chatbot

## Description

PDF Q&A Chatbot is an intelligent assistant that allows users to upload a PDF document and ask questions about its content. Built with Streamlit and powered by advanced language models, this chatbot provides an interactive way to extract information from PDF documents.

Key features:
- PDF upload functionality
- Natural language processing to understand and answer questions
- Context-aware responses based on the PDF content
- Ability to handle follow-up questions and maintain conversation history
- User-friendly interface powered by Streamlit

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/MavinSao/rag-pdf-chatbot.git
   cd pdf-qa-chatbot
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Upload a PDF file using the file uploader.

4. Start asking questions about the content of the PDF in the chat interface.

## Dependencies

- streamlit
- python-dotenv
- pypdf
- langchain
- langchain-core
- langchain-mistralai
- faiss-cpu
- sentence-transformers

For a complete list with versions, see `requirements.txt`.

## Configuration

The chatbot can be configured by modifying the following parameters in `app.py`:

- `chunk_size` and `chunk_overlap` in the `RecursiveCharacterTextSplitter`
- Model parameters in the `ChatMistralAI` initialization
- Memory settings in `ConversationSummaryBufferMemory`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
