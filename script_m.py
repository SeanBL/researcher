import autogen
import os
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config
from memgpt.presets.presets import DEFAULT_PRESET
from memgpt.constants import LLM_MAX_TOKENS
from dotenv import load_dotenv

load_dotenv()
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

config_listM = [
    {
        "model": "gpt-3.5-turbo-1106",  
        # "context_window": LLM_MAX_TOKENS["gpt-3.5-turbo-1106"],
        # "preset": DEFAULT_PRESET,
        # "model_wrapper": None,
        "model_endpoint_type": "openai",
        "model_endpoint": "https://api.openai.com/v1",
        "openai_key": os.getenv("OPENAI_API_KEY"),
    },
]

llm_config = {
    "config_list": config_listM,
    "seed": 42,
}

user_proxy = autogen.UserProxyAgent(
   name="User_Proxy",
   system_message="A user proxy. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
   code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
#    code_execution_config=False,
#    human_input_mode="Always",
#    max_consecutive_auto_reply=1,
#    is_termination_msg=lambda msg: "TERMINATE" in msg["content"]
) 


# Set to True if you want to print MemGPT's inner workings.
# DEBUG = True

# interface_kwargs = {
#     "debug": DEBUG,
#     "show_inner_thoughts": DEBUG,
#     "show_function_outputs": DEBUG,
# }

researcher = create_memgpt_autogen_agent_from_config(
    "MemGPT_agent",
    system_message='''You are a world class reseacher who can do detailed research on any topic and produce fact based results. You do not make things up. You will try as hard as possible to gather facts and data to back up the research. 
        Please make sure you complete the objective above with the following rules:
        1. You should do enough research to gather as much information as possible about the objective.
        2. If there is a URL of relevant links and articles, you will scrape it to gather more information.
        3. After scraping and searching, you should think to yourself, "is there any new things I should search and scrape based on the data I collected to increase research quality?" If the answer is yes, continue, but don't do this more than 3 iterations. 
        4. You should not make things up. You should only write facts and data you have gathered. 
        5. In the final output, you should include all reference data and links to back up your research. You should include all reference data and links to back up your research.
        6. Do not use G2 or Linkedin. They are mostly out dated data. 
        ''',
    llm_config=llm_config,
    #interface_kwargs=interface_kwargs
)

research_manager = autogen.AssistantAgent(
    name="research_manager",
    system_message='''You are a research manager. You are harsh and relentless. You will first try to generate 2 actions a researcher can take to find the information needed. Try to avoid linkedin, or other gated websites that don't allow scraping. You will review the result from the researcher, and always push back if the researcher didn't find the information. Be persistent. For example, if the researcher does not find the correct information, say, "No, you have to find the information. Try again.", and propose another method to try if the researcher can't find an answer. Only after the researcher has found the information will you say, 'TERMINATE'.''',
    llm_config={"config_list": config_list}
)

groupchat = autogen.GroupChat(agents=[user_proxy, researcher, research_manager], messages=[], max_round=10)

manager = autogen.GroupChatManager(
    groupchat=groupchat, 
    llm_config={"config_list": config_list}
)

user_proxy.initiate_chat(
    manager,
    message = "Find articles on the efficacy of an exoskeleton-based physical therapy program for non-ambulatory patients during subacute stroke rehabilitation, and summarize the findings."
)