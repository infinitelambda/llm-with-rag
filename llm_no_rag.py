from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import os
import chainlit as cl

llm_open_ai = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-4o",
    temperature=0,
    streaming=True
)

llm_llama_3 = ChatOllama(model="llama3", temperature=0)

#
selected_model = llm_llama_3

@cl.on_chat_start
async def on_chat_start():
    """
    Asynchronous event handler triggered at the start of a chat session.

    This function initializes the chat model and prompt, setting up a `Runnable`
    object which is then stored in the user session for later use.

    Steps:
        1. Define the chat prompt with predefined messages.
        2. Create a `Runnable` by chaining the prompt, model, and output parser.
        3. Store the `Runnable` in the user session.

    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're an expert data engineer who speaks like a pirate in gaming terms."
                ,
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | selected_model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    """
    Asynchronous event handler triggered when a message is received.

    This function retrieves the `Runnable` object stored in the user session, processes
    the incoming message, streams the response back to the user, and sends the complete
    message.

    Parameters:
        message (cl.Message): The incoming message from the user.

    Steps:
        1. Retrieve the `Runnable` from the user session.
        2. Create a new `cl.Message` object to store the response.
        3. Stream the response tokens from the `Runnable` and update the message content.
        4. Send the complete message to the user.

    """
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
