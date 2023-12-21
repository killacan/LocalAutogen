import autogen
from dotenv import load_dotenv
import os
import memgpt.autogen.memgpt_agent as memgpt_autogen
import memgpt.autogen.interface as autogen_interface 
import memgpt.agent as agent
import memgpt.system as system
import memgpt.utils as utils
import memgpt.presets as presets
import memgpt.constants as constants
import memgpt.personas as personas
import memgpt.humans as humans
from memgpt.persistence_manager import LocalStateManager
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config
import openai

load_dotenv()

openaiApiKey = os.getenv('OPENAI_API_KEY')

# url = "http://127.0.0.1:5000/v1/chat/completions"

config_list = [
    {
        "model": "gpt-3.5-turbo-16k",
        "context_window": 8192,
        "preset": "memgpt_chat",  # NOTE: you can change the preset here
        # OpenAI specific
        "model_endpoint_type": "openai",
        "openai_key": openaiApiKey,
    },
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

interface = autogen_interface.AutoGenInterface()
persistence_manager = LocalStateManager(agent_config=llm_config)
persona = "I am a virtual assistant. I can help you with your tasks."
human = "I'm a team manager and I need to assign tasks to my team members."
memgpt_agent = presets.use_preset(preset.DEFAULT, 'gpt-3.5-turbo-16k', persona, human, persistence_manager=persistence_manager)

interface_kwargs = {
    "debug": False,
    "show_inner_thoughts": True,
    "show_function_outputs": False,
}

coder = create_memgpt_autogen_agent_from_config(
    "MemGPT_agent",
    llm_config=llm_config,
    system_message=f"You are an expert coder, your specialty is writing python code like an expert developer.",
    interface_kwargs=interface_kwargs,
    default_auto_reply="...",
    skip_verify=False,  # NOTE: you should set this to True if you expect your MemGPT AutoGen agent to call a function other than send_message on the first turn
)

pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(agents=[user_proxy, pm, coder], messages=[], max_rounds=12)
manager = autogen.Manager(groupchat=groupchat, llm_config=llm_config)

# task = input("What is the task? ")

user_proxy.initiate_chat(
    assistant, 
    message="first send the message 'lets go mario'",
    )