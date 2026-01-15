#!/usr/bin/env python
# coding: utf-8

# 1. Import LLM
# 2. Import tool
# 3. Bind to LLM
# 4. Prompt template
# 5. custom agent with memory 
# 6. Showcase in gradio

# Import LLM

# In[1]:

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(model="gpt-3.5-turbo", 
                 temperature=0, 
                 openai_api_key=os.environ.get("OPENAI_API_KEY"))


# Import Tool

# In[2]:


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# In[3]:


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


# In[4]:


wikipedia.run("Highest goals in a single season of La Liga")


# Bind Tool with LLM

# In[5]:


tools = [wikipedia]


# In[6]:


llm_with_tools = llm.bind_tools(tools)


# Create Prompt for LLM

# In[7]:


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# In[8]:


from langchain_core.messages import AIMessage, HumanMessage

chat_history = []


# Create Custom Agent with Memory

# In[9]:


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain.agents import create_openai_functions_agent, AgentExecutor


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an epistemic-aware proxy agent."),
    ("human", "{input}"),
])

# In[10]:


from langchain.agents import AgentExecutor

# In[11]:
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

from langchain.callbacks import get_openai_callback


# In[12]:


with get_openai_callback() as cb:
    out = agent_executor.invoke({"input": "Highest goals in a single season of La Liga", "chat_history": chat_history})
print(out)
print(cb)


# In[13]:


chat_history.extend(
    [
        HumanMessage(content="Highest goals in a single season of La Liga"),
        AIMessage(content=out["output"]),
    ]
)

print(chat_history)
agent_executor.invoke({"input": "How many goals he has scored overall in L Liga?", "chat_history": chat_history})


# Showcase in Gradio

# In[14]:


import gradio as gr


# In[15]:


agent_history = []
def call_agent(query, chat_history):
    print("Chat history : ", chat_history)
    output = agent_executor.invoke({"input": query, "chat_history": agent_history})

    agent_history.extend(
    [
        HumanMessage(content="Highest goals in a single season of La Liga"),
        AIMessage(content=out["output"]),
    ]
    )


    chat_history += [
        [
            "<b>Question: </b>" + query,
            "<b>Answer: </b>" + output["output"]
        ]
    ]


    return output["output"], chat_history


# In[16]:


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label = "QnA with Wikipedia")
    question = gr.Textbox(label = "Ask you query here")

    with gr.Row():
        submit = gr.Button("Submit")
        clear = gr.ClearButton([chatbot, question])

    def user(user_message, history):

        bot_message, history = call_agent(user_message, history)

        return "", history

    question.submit(user, [question, chatbot], [question, chatbot], queue=False)
    submit.click(user, [question, chatbot], [question, chatbot], queue=False)

demo.queue()
demo.launch()






