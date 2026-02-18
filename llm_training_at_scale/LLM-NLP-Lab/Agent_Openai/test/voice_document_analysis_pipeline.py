import os
import sys
import logging
import asyncio
import numpy as np
import sounddevice as sd
from typing import List, Dict, Any, Optional
from openai import OpenAI
from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key, Runner, trace
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline
import spacy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set OpenAI API key (replace with your actual key)
set_default_openai_key("YOUR_API_KEY")
client = OpenAI(api_key='YOUR_API_KEY')

# Load spaCy model for text processing (ensure you have downloaded 'en_core_web_sm')
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)

# --- File Handling Functions ---
def upload_file(file_path: str, vector_store_id: str) -> Dict[str, Any]:
    """
    Upload a file to OpenAI and attach it to a vector store.

    Args:
        file_path (str): Path to the file to upload.
        vector_store_id (str): ID of the vector store to attach the file to.

    Returns:
        Dict[str, Any]: Status of the upload operation.
    """
    file_name = os.path.basename(file_path)
    try:
        with open(file_path, 'rb') as file:
            file_response = client.files.create(file=file, purpose="assistants")
            attach_response = client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_response.id
            )
        logger.info(f"Successfully uploaded and attached {file_name} to vector store {vector_store_id}")
        return {"file": file_name, "status": "success"}
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"file": file_name, "status": "failed", "error": f"File not found: {file_path}"}
    except Exception as e:
        logger.error(f"Error uploading {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}

def create_vector_store(store_name: str) -> Dict[str, Any]:
    """
    Create a vector store for document analysis.

    Args:
        store_name (str): Name of the vector store.

    Returns:
        Dict[str, Any]: Details of the created vector store, or empty dict on failure.
    """
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        logger.info(f"Vector store created: {details}")
        return details
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return {}

def upload_user_files(file_paths: List[str], vector_store_id: str) -> List[Dict[str, Any]]:
    """
    Upload multiple user-provided files to a vector store.

    Args:
        file_paths (List[str]): List of file paths to upload.
        vector_store_id (str): ID of the vector store.

    Returns:
        List[Dict[str, Any]]: List of upload statuses for each file.
    """
    if len(file_paths) < 2 or len(file_paths) > 3:
        logger.error("Please provide exactly 2 or 3 files.")
        raise ValueError("Please provide exactly 2 or 3 files.")
    
    upload_results = []
    for file_path in file_paths:
        result = upload_file(file_path, vector_store_id)
        upload_results.append(result)
    return upload_results

# --- Generalized Function Tools ---
@function_tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    Summarize the given text to a specified maximum length.

    Args:
        text (str): Text to summarize.
        max_length (int): Maximum length of the summary.

    Returns:
        str: Summarized text, or error message on failure.
    """
    try:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        summary = " ".join(sentences[:min(len(sentences), 2)])  # Take first 2 sentences
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return f"Error summarizing text: {str(e)}"

@function_tool
def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract top N keywords from the given text.

    Args:
        text (str): Text to analyze.
        top_n (int): Number of keywords to extract.

    Returns:
        List[str]: List of keywords, or empty list on failure.
    """
    try:
        doc = nlp(text)
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop][:top_n]
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []

@function_tool
def calculate_stats(text: str) -> Dict[str, Any]:
    """
    Calculate basic statistics for the given text (e.g., word count, sentence count).

    Args:
        text (str): Text to analyze.

    Returns:
        Dict[str, Any]: Dictionary of statistics, or empty dict on failure.
    """
    try:
        doc = nlp(text)
        stats = {
            "word_count": len([token for token in doc if token.is_alpha]),
            "sentence_count": len(list(doc.sents)),
            "avg_sentence_length": len(doc) / len(list(doc.sants)) if doc.sents else 0
        }
        return stats
    except Exception as e:
        logger.error(f"Error calculating stats: {str(e)}")
        return {}

# --- Agent Definitions ---
file_analysis_agent = Agent(
    name="FileAnalysisAgent",
    instructions=(
        "You analyze the content of uploaded documents using the FileSearchTool. "
        "For summarization, keyword extraction, or statistical analysis, use the provided function tools."
    ),
    tools=[
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"],  # Replace with actual ID
        ),
        summarize_text,
        extract_keywords,
        calculate_stats
    ],
)

web_search_agent = Agent(
    name="WebSearchAgent",
    instructions=(
        "You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query."
    ),
    tools=[WebSearchTool()],
)

function_tool_agent = Agent(
    name="FunctionToolAgent",
    instructions=(
        "You execute generalized function tools (e.g., summarize_text, extract_keywords, calculate_stats) "
        "based on user queries."
    ),
    tools=[summarize_text, extract_keywords, calculate_stats],
)

triage_agent = Agent(
    name="TriageAgent",
    instructions=prompt_with_handoff_instructions("""
You are an advanced virtual assistant for real-time document analysis.
Welcome the user and ask how you can help. Based on the user's intent, route to:
- FileAnalysisAgent for queries about document content.
- WebSearchAgent for queries requiring real-time web search.
- FunctionToolAgent for queries involving text summarization, keyword extraction, or statistical analysis.
"""),
    handoffs=[file_analysis_agent, web_search_agent, function_tool_agent],
)

# --- Voice Assistant ---
async def voice_assistant(vector_store_id: str):
    """
    Run a voice-based assistant for real-time document analysis.

    Args:
        vector_store_id (str): ID of the vector store containing uploaded files.
    """
    samplerate = sd.query_devices(kind='input')['default_samplerate']
    logger.info("Starting voice assistant. Press Enter to speak, or type 'esc' to exit.")

    while True:
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(triage_agent))

        try:
            cmd = input("Press Enter to speak your query (or type 'esc' to exit): ")
            if cmd.lower() == "esc":
                logger.info("Exiting voice assistant...")
                break

            logger.info("Listening...")
            recorded_chunks = []

            with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16',
                               callback=lambda indata, frames, time, status: recorded_chunks.append(indata.copy())):
                input()

            recording = np.concatenate(recorded_chunks, axis=0)
            audio_input = AudioInput(buffer=recording)

            with trace("Document Analysis Voice Assistant"):
                result = await pipeline.run(audio_input)

            response_chunks = []
            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    response_chunks.append(event.data)

            response_audio = np.concatenate(response_chunks, axis=0)

            logger.info("Assistant is responding...")
            sd.play(response_audio, samplerate=samplerate)
            sd.wait()
            logger.info("Response completed.")
        except Exception as e:
            logger.error(f"Error in voice assistant: {str(e)}")

# --- CLI Interface ---
async def cli_interface(vector_store_id: str):
    """
    Run a command-line interface for text-based queries.

    Args:
        vector_store_id (str): ID of the vector store containing uploaded files.
    """
    logger.info("Starting CLI interface. Type 'exit' to quit, or 'voice' to switch to voice mode.")
    while True:
        try:
            query = input("Enter your query (or type 'exit' to quit, 'voice' to switch to voice mode): ")
            if query.lower() == "exit":
                logger.info("Exiting CLI interface...")
                break
            elif query.lower() == "voice":
                logger.info("Switching to voice mode...")
                await voice_assistant(vector_store_id)
                continue

            with trace("Document Analysis CLI"):
                result = await Runner.run(triage_agent, query)
                print(f"Assistant: {result.final_output}")
                print("---")
        except Exception as e:
            logger.error(f"Error in CLI interface: {str(e)}")

# --- Main Execution ---
async def main():
    """
    Main function to set up and run the document analysis pipeline.
    """
    try:
        # Create a vector store
        vector_store = create_vector_store("DocumentAnalysisStore")
        if not vector_store:
            logger.error("Failed to create vector store. Exiting...")
            sys.exit(1)

        vector_store_id = vector_store["id"]

        # Get user input for files
        print("Please provide paths to 2 or 3 files for analysis (one per line, press Enter twice to finish):")
        file_paths = []
        while len(file_paths) < 3:
            file_path = input().strip()
            if not file_path:
                break
            file_paths.append(file_path)

        # Upload files
        upload_results = upload_user_files(file_paths, vector_store_id)
        for result in upload_results:
            if result["status"] == "success":
                logger.info(f"Successfully uploaded {result['file']}")
            else:
                logger.error(f"Failed to upload {result['file']}: {result['error']}")

        # Update FileSearchTool with the correct vector store ID
        file_analysis_agent.tools[0].vector_store_ids = [vector_store_id]

        # Start CLI interface
        await cli_interface(vector_store_id)
    except ValueError as ve:
        logger.error(str(ve))
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())