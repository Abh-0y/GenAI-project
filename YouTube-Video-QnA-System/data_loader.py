"""Module to handle YouTube transcript ingestion."""

from typing import List
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document

def fetch_youtube_documents(video_id: str, languages: List[str] = None) -> List[Document]:
    """
    Fetches transcript segments from YouTube and maps them to LangChain Document objects.
    """
    if languages is None:
        languages = ["en"]
        
    transcript_chunks = YouTubeTranscriptApi().fetch(video_id=video_id, languages=languages)
    
    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "video_id": video_id,
                "start": chunk.start,
                "duration": chunk.duration
            }
        )
        for chunk in transcript_chunks
    ]
    return documents