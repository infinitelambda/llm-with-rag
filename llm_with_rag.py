import os
import chainlit as cl
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
import document_processor

chat_open_ai = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-4o",
    temperature=0,
    streaming=True
)

llm_llama_3 = ChatOllama(model="llama3", temperature=0)

selected_model = llm_llama_3

@cl.on_chat_start
async def main():
    """
    Asynchronous event handler triggered at the start of a chat session.

    This function performs the following steps:
        1. Prompts the user to upload a file.
        2. Processes the uploaded file and sends a message indicating the file is being processed.
        3. Initializes a document search object from the processed file.
        4. Defines a prompt template for generating answers from document sections.
        5. Sets up a retrieval-based QA chain using the specified LLM and the document search retriever.
        6. Sends a message indicating that the file has been processed and the user can now ask questions.
        7. Stores the QA chain in the user session.

    The prompt template used ensures that the system does not fabricate answers and indicates when the answer is unknown.
    """
    file = await document_processor.get_file_from_user()

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    docsearch = await cl.make_async(document_processor.get_docsearch)(file)
    prompt_template = """Use the following pieces of context to answer the question at the end . 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always include the sources you find relevant to the question.
        
    {summaries}
    Question: {question}
    Helpful Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question", "summaries"]
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=selected_model,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": PROMPT
        },
        retriever=docsearch.as_retriever(max_tokens_limit=4097),
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    """
    Asynchronous event handler triggered when a message is received.

    This function performs the following steps:
        1. Retrieves the user's input message.
        2. Retrieves the QA chain from the user session.
        3. Sets up a callback handler for processing the final answer.
        4. Executes the QA chain with the user's input prompt.
        5. Retrieves and processes the answer and sources from the chain response.
        6. Matches sources with the original documents and creates text elements for the response.
        7. Constructs and sends the final response message with the answer and sources.

    Parameters:
        message (cl.Message): The incoming message from the user containing the question.
    """
    prompt = message.content

    chain = cl.user_session.get("chain")
    langchain_callback = cl.AsyncLangchainCallbackHandler()
    langchain_callback.answer_reached = True
    res = await chain.acall(prompt, callbacks=[langchain_callback])
    print(res)
    answer = res["answer"]
    sources = res["sources"]

    source_elements = []

    # Get the documents from the user session
    docs = cl.user_session.get("docs")
    docs_metadata = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in docs_metadata]

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            source_links = ', '.join([f"[{name}](#{name})" for name in found_sources])
            answer += f"\nSources: {source_links}"
        else:
            answer += "\nNo sources found"

    if langchain_callback.has_streamed_final_answer:
        langchain_callback.final_stream.elements = source_elements
        await langchain_callback.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()
