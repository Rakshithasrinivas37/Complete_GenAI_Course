from crewai import Task
from tools import YouTubeTool
from agents import blog_researcher, blog_writer

yt_tool = YouTubeTool()

## Research task
research_task = Task(
    description=(
        "Identify the video {topic}."
        "Get detailed information about the video from the channel"
    ),
    expected_output='A comprehensive 3 paragraphs long report based on the {topic} of video content',
    tools=[yt_tool],
    agent=blog_researcher
)

writing_task = Task(
    description="Get the info from the youtube channel on the topic {topic}",
    expected_output='Summarize the info from the youtube channel video on the topic{topic} and create content for the blog',
    tools=[yt_tool],
    agent=blog_writer,
    async_execution=False, ##if this parameter is true, then both the agents will work in parallel -> Now it is sequential
    output_file='blog_post.md'
)

