"""CLI entry point for querying the YouTube video transcript."""

import config
from data_loader import fetch_youtube_documents
from rag_pipeline import build_vector_store, create_rag_chain

def main():
    video_id = input(f"Enter YouTube Video ID (Press Enter for default: {config.DEFAULT_VIDEO_ID}): ").strip()
    if not video_id:
        video_id = config.DEFAULT_VIDEO_ID

    print(f"Fetching transcript for video: {video_id}...")
    docs = fetch_youtube_documents(video_id, config.LANGUAGE_CODES)
    print(f"Ingested {len(docs)} transcript segments.")

    print("Building FAISS index...")
    vector_store = build_vector_store(docs)
    qa_pipeline = create_rag_chain(vector_store)
    
    print("\nReady! Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question about the video: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue
            
        response = qa_pipeline(query)
        print(f"\nResponse:\n{response.content}")

if __name__ == "__main__":
    main()