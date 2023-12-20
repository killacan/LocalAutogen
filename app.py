import autogen
from dotenv import load_dotenv
import os

load_dotenv()

openAiApiKey = os.getenv('OPENAI_API_KEY')

config_list = [
    {
        'model': 'gpt-3.5-turbo-16k',
        'api_key' : openAiApiKey,
    }
]

llm_config={
    "timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "web"},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction, otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)

task = input("What is the task? ")

user_proxy.initiate_chat(
    assistant, 
    message=task
    )