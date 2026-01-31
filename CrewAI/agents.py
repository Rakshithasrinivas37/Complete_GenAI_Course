from crewai import Agent
from tools import YouTubeTool
from crewai import LLM

import os
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="groq/qwen/qwen3-32b",
    temperature=0.7
)

yt_tool = YouTubeTool()

##Create a senior blog content researcher agent
blog_researcher = Agent(
    llm=llm,
    role='Blog Researcher for youtube videos',
    goal='Get the relevant content for the topic{topic} from Youtube channel in 300 words',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data science, Machine learning and Gen AI, and providing suggestions"
    ),
    tools=[yt_tool],
    allow_delegations=True
)

## Create a blog writer agent with YT tool
blog_writer = Agent(
    llm=llm,
    role='Blog writer',
    goal='Narrate compelling tech stories about the video{topic} from Youtube channel in 300 words',
    verbose=True,
    memory=True,
    backstory=(
        """
        With a flair for simplifying complex topics,
        you craft engaging narratives that captivate and educate,
        bringing new discoveries to light in an accessibkle manner. 
    """
    ),
    tools=[yt_tool],
    allow_delegations=False
)





