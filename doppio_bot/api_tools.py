import frappe

from langchain import OpenAI
from langchain.agents import tool
from langchain.agents import AgentType
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from langchain.agents import initialize_agent

from .api import get_model_from_settings

@frappe.whitelist()
def todo_response(session_id: str, prompt_message: str):
    # Throw if no key in site_config
    # Maybe extract and cache this (site cache)
    opeai_api_key = frappe.conf.get("openai_api_key")
    openai_model = get_model_from_settings()

    if not opeai_api_key:
        frappe.throw("Please set `openai_api_key` in site config")

    llm = OpenAI(model_name=openai_model, temperature=0, openai_api_key=opeai_api_key)

    message_history = RedisChatMessageHistory(
        session_id=session_id,
        url=frappe.conf.get("redis_cache") or "redis://localhost:6379/0",
    )
    memory = ConversationBufferMemory(memory_key="history", chat_memory=message_history)
    # memory = ConversationBufferMemory(memory_key="chat_history")
    tools = [create_todo]

    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
    )

    # Will call the tool with proper JSON and voila, magic!
    agent_chain.run("I have to create a college report before May 17, 2023, can you set a task for me?")

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
        return f"He creado una nueva tarea llamada {todo.name}, \nencuentrala en http://chatgpt.localhost:8000/app/todo/{todo.name}"
    except Exception:
        return "failed"
