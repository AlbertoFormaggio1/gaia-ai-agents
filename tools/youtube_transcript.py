from langchain_core.tools.base import BaseTool, ToolException
import requests
from youtube_transcript_api import YouTubeTranscriptApi
import re

class YoutubeTranscriptTool(BaseTool):
    name: str = "youtube_transcript_tool"
    description: str = "This tool can be used to retrieve the transcript of a youtube video given the FULL youtube link. You must pass the full youtube link!"

    def _run(self, youtube_link: str) -> str:
        """
        Fetch transcript for a YouTube video URL.
        Args:
            youtube_link: The full URL of the YouTube video.
        Returns:
            The transcript as a single string.
        """
        # Get the video ID from the youtube URL
        re_match = re.search(r"watch\?v=([^&]+)", youtube_link)
        if not re_match:
            raise ValueError(f"Invalid YouTube URL: {youtube_link}")
        video_id = re_match.group(1)

        # Initialize the transcriptAPI and retrieve the transcript for the given videoID
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)

        transcript = []
        for snippet in fetched_transcript:
            transcript.append(snippet.text)

        return "\n".join(transcript)