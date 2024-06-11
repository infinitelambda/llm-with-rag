from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from chainlit.types import AskFileResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import chainlit as cl

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
embeddings = OpenAIEmbeddings()
text_splitter = SemanticChunker(embeddings)


def process_file(file: AskFileResponse):
    """
    Processes the uploaded file by splitting its content into smaller chunks.

    The function determines the file type (text or PDF), loads the file content,
    and then splits the content into smaller chunks using a text splitter.

    Parameters:
        file (AskFileResponse): The file uploaded by the user.

    Returns:
        List[Document]: A list of processed document chunks with updated metadata.
    """
    import tempfile

    if file.type == "text/plain":
        loader = TextLoader
    elif file.type == "application/pdf":
        loader = PyPDFLoader

    with tempfile.NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)
        loader = loader(tempfile.name)
        documents = loader.load()
        documents_split = text_splitter.split_documents(documents)

        for i, doc in enumerate(documents_split):
            doc.metadata["source"] = f"source_{i}"

        return documents_split


def get_docsearch(file: AskFileResponse):
    """
    Creates a document search index from the processed file.

    This function processes the uploaded file, stores the processed document chunks
    in the user session, and creates a Chroma search index using the document chunks
    and embeddings.

    Parameters:
        file (AskFileResponse): The file uploaded by the user.

    Returns:
        Chroma: A Chroma search index created from the processed document chunks.
    """
    docs = process_file(file)

    cl.user_session.set("docs", docs)

    return Chroma.from_documents(docs, embeddings)


welcome_message = """Welcome to DocQuery! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""


async def get_file_from_user():
    """
    Asynchronously prompts the user to upload a file and waits for the file upload.

    The function sends a message asking the user to upload a PDF or text file,
    and waits until the user uploads a file or the request times out.

    Returns:
        AskFileResponse: The file uploaded by the user.
    """
    await cl.Message(content="Hello! Please upload a pdf in order to ask questions about it.").send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    return files[0]
