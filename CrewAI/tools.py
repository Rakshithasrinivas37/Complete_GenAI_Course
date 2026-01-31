from crewai.tools import BaseTool
from langchain_community.tools import YouTubeSearchTool

class YouTubeTool(BaseTool):
    name: str = "YouTube Search"
    description: str = "Search YouTube videos for a given topic"

    def _run(self, query: str):
        yt = YouTubeSearchTool()
        return yt.run(query)
