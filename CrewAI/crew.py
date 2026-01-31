from crewai import Crew, Process
from agents import blog_writer, blog_researcher
from tasks import research_task, writing_task

crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, writing_task],
    process=Process.sequential, ## for sequential task execution
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

## To start the execution
result = crew.kickoff(inputs={'topic': 'AI vs ML vs DL vs Data Science'})
print(result)