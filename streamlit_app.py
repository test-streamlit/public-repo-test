#!/usr/bin/env python3
"""
Streamlit Web App for Social Media Agent

A web interface for the social media automation and management tool.
"""

import streamlit as st
import asyncio
import os
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from agents import Agent, Runner, WebSearchTool, function_tool, ItemHelpers, trace
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List
import time

# ---------------------------------------------------------------------------------------
# Step 0: Load environment variables
# ---------------------------------------------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in the environment variables")
    st.stop()

# ---------------------------------------------------------------------------------------
# Step 1: Define Tools for agents (same as original)
# ---------------------------------------------------------------------------------------

@function_tool
def generate_content(video_transcript: str, social_media_platform: str) -> str:
    """Generate social media content from a transcript."""
    print(f"[INFO] Generating social media content for {social_media_platform} from transcript using the generate_content tool")

    # Initialize the OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Generate the social media content
    response = client.responses.create(
        model="gpt-4o",
        input=f"Here is the transcript of a YouTube video: {video_transcript}. "
               f"Generate social media content for {social_media_platform} from transcript.",
        max_output_tokens=2500,
    )

    return response.output_text

# ---------------------------------------------------------------------------------------
# Step 2: Define Agents (same as original)
# ---------------------------------------------------------------------------------------

@dataclass
class Posts:
    platform: str
    content: str

content_writer_agent = Agent(
    name="Content Writer Agent",
    instructions="""You are a talented content writer agent who writes engaging, humorous, and informative highly readable content for social media platforms.
                 You are given a video transcript and social media platforms. 
                 You need to generate social media content from the transcript using the generate_content tool for each of the given platforms.
                 You can use the websearch tool to find relevant information on the topic and fill in some useful details if needed.""",
    model="gpt-4o-mini",
    tools=[
        generate_content,
        WebSearchTool()
        ],
    output_type=List[Posts],
)

# ---------------------------------------------------------------------------------------
# Step 3: Helper functions (same as original)
# ---------------------------------------------------------------------------------------

def get_video_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetch the video transcript from a YouTube video using the video id.
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        str: The concatenated transcript text
        
    Raises:
        ValueError: If video_id is empty or invalid
        Exception: For other API-related errors
    """
    print(f"[INFO] Fetching transcript for video ID: {video_id} in language: {language} using YouTubeTranscriptApi")
    
    # Input validation
    if not video_id or not video_id.strip():
        raise ValueError("Video ID cannot be empty or None")
    
    # Clean the video ID (remove any URL parts if accidentally included)
    video_id = video_id.strip()
    if "youtube.com" in video_id or "youtu.be" in video_id:
        # Extract just the video ID from URL
        if "v=" in video_id:
            video_id = video_id.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_id:
            video_id = video_id.split("youtu.be/")[1].split("?")[0]
    
    if language is None:
        language = "en"

    try:
        # Use the new API: create instance and fetch
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=[language])
        
        # The new API returns a FetchedTranscript object that's iterable
        # Extract text from each snippet
        transcript_text = " ".join(snippet.text for snippet in fetched_transcript)
        
        return transcript_text.strip()
        
    except Exception as e:
        # Handle different types of errors with specific messages
        if isinstance(e, ValueError):
            error_msg = f"Invalid video ID format: '{video_id}'. Please provide a valid YouTube video ID."
            print(f"ValueError: {error_msg}")
            raise ValueError(error_msg) from e
            
        elif isinstance(e, KeyError):
            error_msg = f"Video '{video_id}' not found or is private/deleted."
            print(f"KeyError: {error_msg}")
            raise KeyError(error_msg) from e
            
        elif isinstance(e, ConnectionError):
            error_msg = f"Network connection error while fetching transcript for video '{video_id}'. Please check your internet connection."
            print(f"ConnectionError: {error_msg}")
            raise ConnectionError(error_msg) from e
            
        elif isinstance(e, TimeoutError):
            error_msg = f"Request timeout while fetching transcript for video '{video_id}'. The request took too long to complete."
            print(f"TimeoutError: {error_msg}")
            raise TimeoutError(error_msg) from e
            
        elif isinstance(e, PermissionError):
            error_msg = f"Access denied for video '{video_id}'. The video may be private, restricted, or require authentication."
            print(f"PermissionError: {error_msg}")
            raise PermissionError(error_msg) from e
            
        else:
            # Handle any other unexpected errors
            error_msg = f"Unexpected error while fetching transcript for video '{video_id}': {str(e)}"
            print(f"Exception: {error_msg}")
            raise Exception(error_msg) from e

# ---------------------------------------------------------------------------------------
# Step 4: Streamlit UI
# ---------------------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Social Media Content Generator",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("📱 Social Media Content Generator")
    st.markdown("Generate engaging social media content from YouTube video transcripts using AI!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Language selection
        language = st.selectbox(
            "Transcript Language",
            ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            index=0,
            help="Select the language of the video transcript"
        )
        
        # Platform selection
        st.subheader("📋 Target Platforms")
        platforms = st.multiselect(
            "Select platforms for content generation",
            ["LinkedIn", "Instagram", "Twitter", "Facebook", "TikTok", "YouTube"],
            default=["LinkedIn", "Instagram"],
            help="Choose which social media platforms to generate content for"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - **Video ID**: Extract from YouTube URL (e.g., `dQw4w9WgXcQ` from `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
        - **Query**: Be specific about what type of content you want
        - **Platforms**: Different platforms have different content requirements
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎥 Video Information")
        
        # Video ID input
        video_id = st.text_input(
            "YouTube Video ID or URL",
            placeholder="e.g., dQw4w9WgXcQ or https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            help="Enter the YouTube video ID or full URL"
        )
        
        # Query input
        query = st.text_area(
            "Content Generation Query",
            placeholder="e.g., Generate a LinkedIn post and Instagram caption that highlights the key insights from this video...",
            height=100,
            help="Describe what type of content you want to generate"
        )
        
        # Generate button
        if st.button("🚀 Generate Content", type="primary", use_container_width=True):
            if not video_id.strip():
                st.error("Please enter a video ID or URL")
            elif not query.strip():
                st.error("Please enter a query")
            elif not platforms:
                st.error("Please select at least one platform")
            else:
                # Show progress
                with st.spinner("🔄 Processing..."):
                    try:
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Fetch transcript
                        status_text.text("📥 Fetching video transcript...")
                        progress_bar.progress(25)
                        
                        transcript = get_video_transcript(video_id, language)
                        
                        # Step 2: Generate content
                        status_text.text("🤖 Generating social media content...")
                        progress_bar.progress(75)
                        
                        # Prepare the message for the agent
                        platform_list = ", ".join(platforms)
                        msg = f"{query} Generate content for these platforms: {platform_list}. Video transcript: {transcript}"
                        
                        input_data = [{"role": "user", "content": msg}]
                        
                        # Run the agent
                        with trace("Writing social media content"):
                            result = asyncio.run(Runner.run(content_writer_agent, input_data))
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Content generated successfully!")
                        
                        # Store results in session state
                        st.session_state.results = result.final_output
                        st.session_state.transcript = transcript
                        st.session_state.video_id = video_id
                        
                        time.sleep(1)  # Brief pause to show completion
                        progress_bar.empty()
                        status_text.empty()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.stop()
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        if 'results' in st.session_state:
            st.metric("Platforms", len(st.session_state.results))
            st.metric("Video ID", st.session_state.video_id[:10] + "..." if len(st.session_state.video_id) > 10 else st.session_state.video_id)
            
            # Transcript preview
            if 'transcript' in st.session_state:
                with st.expander("📝 Transcript Preview"):
                    st.text(st.session_state.transcript[:200] + "..." if len(st.session_state.transcript) > 200 else st.session_state.transcript)
    
    # Display results
    if 'results' in st.session_state and st.session_state.results:
        st.markdown("---")
        st.subheader("🎯 Generated Content")
        
        # Create tabs for each platform
        if len(st.session_state.results) > 1:
            tab_names = [post.platform for post in st.session_state.results]
            tabs = st.tabs(tab_names)
            
            for i, (tab, post) in enumerate(zip(tabs, st.session_state.results)):
                with tab:
                    st.markdown(f"### {post.platform}")
                    
                    # Copy button
                    if st.button(f"📋 Copy {post.platform} Content", key=f"copy_{i}"):
                        st.write("Content copied to clipboard!")
                        st.code(post.content, language="text")
                    else:
                        st.text_area(
                            f"{post.platform} Content",
                            value=post.content,
                            height=300,
                            key=f"content_{i}",
                            disabled=True
                        )
        else:
            # Single result
            post = st.session_state.results[0]
            st.markdown(f"### {post.platform}")
            
            if st.button("📋 Copy Content"):
                st.write("Content copied to clipboard!")
                st.code(post.content, language="text")
            else:
                st.text_area(
                    f"{post.platform} Content",
                    value=post.content,
                    height=400,
                    disabled=True
                )
        
        # Download option
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Download as Text", use_container_width=True):
                content_text = "\n\n".join([f"=== {post.platform} ===\n{post.content}" for post in st.session_state.results])
                st.download_button(
                    label="📥 Download",
                    data=content_text,
                    file_name=f"social_media_content_{st.session_state.video_id}.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("🔄 Generate New Content", use_container_width=True):
                # Clear session state
                for key in ['results', 'transcript', 'video_id']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main() 