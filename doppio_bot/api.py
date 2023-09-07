import frappe
import json
import os

import openai

from langchain.llms import OpenAI
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

from langchain.agents import Tool
from langchain.agents import tool
from langchain.agents import AgentType

from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

# Note: Copied the default template and added extra instructions for code output
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    output_parser=None,
    partial_variables={},
    template="""
    The following is a friendly conversation between a human and an AI.
    The AI is talkative and provides lots of specific details from its context. The AI's name is DoppioBot and it's birth date it 24th April, 2023.
    If the AI does not know the answer to a question, it truthfully says it does not know.
    Any programming code should be output in a github flavored markdown code block mentioning the programming language.


    Current conversation:
    {history}
    Human: {input}
    AI:""",
    template_format="f-string",
    validate_template=True,
)


@frappe.whitelist()
def get_chatbot_response(session_id: str, prompt_message: str) -> str:
    # Throw if no key in site_config
    # Maybe extract and cache this (site cache)
    opeai_api_key = frappe.conf.get("openai_api_key")
    serpapi_api_key = frappe.conf.get("serpapi_api_key")
    openai_model = get_model_from_settings()

    if not opeai_api_key:
        frappe.throw("Please set `openai_api_key` in site config")

    if not serpapi_api_key:
        frappe.throw("Please set `serpapi_api_key` in site config")

    os.environ['SERPAPI_API_KEY'] = serpapi_api_key
    os.environ['OPENAI_API_KEY'] = opeai_api_key

    # openai.api_type = os.getenv('OPENAI_API_TYPE')
    # openai.api_version = os.getenv('OPENAI_API_VERSION')
    # openai.api_base = os.getenv('OPENAI_API_BASE')
    # openai.api_key = os.getenv('OPENAI_API_KEY')

    # Init langchain tools
    params = {
        "engine": "bing",
        "cc": "US",
        "api_key": serpapi_api_key
    }

    search = SerpAPIWrapper(params=params)
    tools = [
        Tool(
            name        = "Current Search",
            func        = search.run,
            description = "useful for when you need to answer question about current events or the current state of the world"
        ),
        create_todo
    ]

    # llm = OpenAI(model_name=openai_model, temperature=0, openai_api_key=opeai_api_key)
    llm = OpenAI(temperature=0)

    message_history = RedisChatMessageHistory(
        session_id=session_id,
        url=frappe.conf.get("redis_cache") or "redis://localhost:6379/0",
    )

    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, prompt_template=prompt_template)
    response = agent_chain.run(input=prompt_message)

    # memory = ConversationBufferMemory(memory_key="history", chat_memory=message_history)
    # conversation_chain = ConversationChain(llm=llm, memory=memory, prompt=prompt_template)
    # response = conversation_chain.run(prompt_message)
    return response


def get_model_from_settings():
    return (
        frappe.db.get_single_value("DoppioBot Settings", "openai_model") or "gpt-3.5-turbo"
    )

@tool
def create_todo(todo: str) -> str:
    """
    Create a new ToDo document, can be used when you need to store a note or todo or task for the user.
    It takes a json string as input and requires at least the `description`. Returns "done" if the
    todo was created and "failed" if the creation failed. Optionally it could contain a `date`
    field (in the JSON) which is the due date or reminder date for the task or todo. The `date` must follow
    the "YYYY-MM-DD" format. You don't need to add timezone to the date.
    """
    try:
        data = frappe.parse_json(todo)
        todo = frappe.new_doc("ToDo")
        todo.update(data)
        todo.save()
        return "done"
    except Exception:
        return "failed"
