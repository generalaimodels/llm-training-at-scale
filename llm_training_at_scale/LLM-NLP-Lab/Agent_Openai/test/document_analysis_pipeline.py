# Standard library imports
import os
import sys
import asyncio
from typing import List, Dict, Union, Optional
from datetime import datetime

# Third-party imports
from openai import OpenAI, OpenAIError
from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key, Runner, trace
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# Set OpenAI API key (replace with your actual key or use environment variable for security)
set_default_openai_key(os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"))

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"))


# --- Generalized Function Tools ---
@function_tool
def extract_keywords(text: str, max_keywords: int = 5) -> Dict[str, Union[List[str], str]]:
    """
    Extract key topics or keywords from a given text using a simple heuristic or NLP model.
    This is a generalized tool for document analysis.

    Args:
        text (str): The text to analyze.
        max_keywords (int): Maximum number of keywords to extract.

    Returns:
        Dict[str, Union[List[str], str]]: Dictionary containing keywords or error message.
    """
    try:
        if not text or not isinstance(text, str):
            return {"status": "error", "message": "Invalid input text"}

        # Placeholder for keyword extraction logic (can be enhanced with NLP models like spaCy or BERT)
        words = text.lower().split()
        keywords = list(set([word for word in words if len(word) > 3 and word.isalpha()]))[:max_keywords]
        return {"status": "success", "keywords": keywords}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@function_tool
def summarize_text(text: str, max_length: int = 100) -> Dict[str, str]:
    """
    Summarize a given text to a specified length (in words).

    Args:
        text (str): The text to summarize.
        max_length (int): Maximum length of the summary in words.

    Returns:
        Dict[str, str]: Dictionary containing summary or error message.
    """
    try:
        if not text or not isinstance(text, str):
            return {"status": "error", "message": "Invalid input text"}

        words = text.split()
        if len(words) <= max_length:
            return {"status": "success", "summary": text}
        
        summary = " ".join(words[:max_length]) + "..."
        return {"status": "success", "summary": summary}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@function_tool
def compare_documents(doc1: str, doc2: str) -> Dict[str, Union[str, float]]:
    """
    Compare two documents for similarity (using simple word overlap as a placeholder).

    Args:
        doc1 (str): First document text.
        doc2 (str): Second document text.

    Returns:
        Dict[str, Union[str, float]]: Dictionary containing similarity score or error message.
    """
    try:
        if not doc1 or not doc2 or not isinstance(doc1, str) or not isinstance(doc2, str):
            return {"status": "error", "message": "Invalid input documents"}

        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())
        common_words = words1.intersection(words2)
        total_words = words1.union(words2)
        similarity = len(common_words) / len(total_words) if total_words else 0.0
        return {"status": "success", "similarity": similarity}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- File and Vector Store Management ---
def upload_file(file_path: str, vector_store_id: str) -> Dict[str, str]:
    """
    Upload a file to OpenAI and attach it to a vector store for retrieval.

    Args:
        file_path (str): Path to the file to upload.
        vector_store_id (str): ID of the vector store to attach the file to.

    Returns:
        Dict[str, str]: Dictionary with file upload status and details.
    """
    file_name = os.path.basename(file_path)
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        file_response = client.files.create(file=open(file_path, 'rb'), purpose="assistants")
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        return {"file": file_name, "status": "success"}
    except FileNotFoundError as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}
    except OpenAIError as e:
        print(f"OpenAI API error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}
    except Exception as e:
        print(f"Unexpected error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}


def create_vector_store(store_name: str) -> Dict[str, Union[str, int]]:
    """
    Create a vector store for storing document embeddings.

    Args:
        store_name (str): Name of the vector store.

    Returns:
        Dict[str, Union[str, int]]: Dictionary with vector store details or empty dict on error.
    """
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print("Vector store created:", details)
        return details
    except OpenAIError as e:
        print(f"OpenAI API error creating vector store: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error creating vector store: {e}")
        return {}


def read_file_content(file_path: str) -> str:
    """
    Read the content of a file (supports text files).

    Args:
        file_path (str): Path to the file.

    Returns:
        str: File content or empty string on error.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return ""
    except UnicodeDecodeError as e:
        print(f"Error decoding file {file_path}: {str(e)}")
        return ""
    except Exception as e:
        print(f"Unexpected error reading file {file_path}: {str(e)}")
        return ""


# --- Agents ---
# Agent 1: Document Analysis Agent
doc_analysis_agent = Agent(
    name="DocumentAnalysisAgent",
    instructions=(
        "You perform advanced analysis on documents, including keyword extraction, summarization, "
        "and document comparison. Use the appropriate tools to fulfill user queries."
    ),
    tools=[extract_keywords, summarize_text, compare_documents],
)

# Agent 2: File Search Agent
file_search_agent = Agent(
    name="FileSearchAgent",
    instructions=(
        "You search and retrieve information from uploaded files using the FileSearchTool. "
        "Provide concise and accurate answers based on the file content."
    ),
    tools=[FileSearchTool(
        max_num_results=3,
        vector_store_ids=["VECTOR_STORE_ID"],  # Replace with actual vector store ID
    )],
)

# Agent 3: Web Search Agent
web_search_agent = Agent(
    name="WebSearchAgent",
    instructions=(
        "You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query."
    ),
    tools=[WebSearchTool()],
)

# Agent 4: Triage Agent (Main Agent for Routing)
triage_agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions("""
You are the virtual assistant for real-time document analysis. Welcome the user and ask how you can help.
Based on the user's intent, route to:
- DocumentAnalysisAgent for document analysis tasks (e.g., summarization, keyword extraction, comparison)
- FileSearchAgent for queries about uploaded file content
- WebSearchAgent for anything requiring real-time web search
If the query is unclear, ask for clarification.
"""),
    handoffs=[doc_analysis_agent, file_search_agent, web_search_agent],
)


# --- Command-Line Interface ---
def get_user_files() -> List[str]:
    """
    Prompt the user to input file paths and validate them.

    Returns:
        List[str]: List of valid file paths.
    """
    print("Please enter the paths to 2 or 3 files for analysis (one per line). Press Enter twice to finish.")
    file_paths = []
    while len(file_paths) < 3:
        file_path = input(f"Enter file path {len(file_paths) + 1} (or press Enter to finish): ").strip()
        if not file_path:
            if len(file_paths) < 2:
                print("You must provide at least 2 files.")
                continue
            break
        if os.path.exists(file_path):
            file_paths.append(file_path)
        else:
            print(f"Error: File '{file_path}' does not exist. Please try again.")
    return file_paths


async def process_user_query(query: str) -> None:
    """
    Process a single user query using the triage agent.

    Args:
        query (str): The user's query.
    """
    try:
        with trace("Document Analysis Assistant"):
            result = await Runner.run(triage_agent, query)
            print(f"User: {query}")
            print(result.final_output)
            print("---")
    except Exception as e:
        print(f"Error processing query '{query}': {str(e)}")


async def main():
    """
    Main function to run the pipeline, including file uploads and user query interface.
    """
    print("Welcome to the Real-Time Document Analysis Pipeline!")
    
    # Step 1: Get user files
    file_paths = get_user_files()
    if not file_paths:
        print("No valid files provided. Exiting.")
        sys.exit(1)

    # Step 2: Create a vector store for the files
    vector_store_name = f"DocumentAnalysisStore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vector_store = create_vector_store(vector_store_name)
    if not vector_store:
        print("Failed to create vector store. Exiting.")
        sys.exit(1)

    vector_store_id = vector_store["id"]

    # Step 3: Upload files to the vector store
    print("\nUploading files to vector store...")
    for file_path in file_paths:
        upload_result = upload_file(file_path, vector_store_id)
        print(f"Upload result for {upload_result['file']}: {upload_result['status']}")

    # Step 4: Interactive query loop
    print("\nFiles have been uploaded and are ready for analysis.")
    print("You can now ask questions about the files, request document analysis, or perform web searches.")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            print("Please enter a valid query.")
            continue
        await process_user_query(query)


# --- Example Usage ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")