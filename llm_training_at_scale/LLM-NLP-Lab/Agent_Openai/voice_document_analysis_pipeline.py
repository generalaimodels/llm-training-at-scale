"""
Advanced Document Analysis Pipeline with Multi-Agent Architecture
This system provides real-time analysis of documents using specialized agents,
various function tools, and voice interaction capabilities.
"""

import os
import re
import json
import time
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
import asyncio
import datetime
from dataclasses import dataclass
from urllib.parse import urlparse

# OpenAI and Agent Framework
from openai import OpenAI
from agents import Agent, Runner, function_tool, WebSearchTool, FileSearchTool, trace
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline, TextToSpeechService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("document_analysis_pipeline")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))

# Document and Vector Store Management
class DocumentManager:
    """Manages document processing, vector stores, and file operations."""
    
    def __init__(self):
        self.vector_stores = {}
        self.uploaded_files = {}
    
    def create_vector_store(self, store_name: str) -> Dict[str, Any]:
        """Create a new vector store for document embeddings."""
        try:
            vector_store = client.vector_stores.create(name=store_name)
            details = {
                "id": vector_store.id,
                "name": vector_store.name,
                "created_at": vector_store.created_at,
                "file_count": vector_store.file_counts.completed
            }
            self.vector_stores[store_name] = details
            logger.info(f"Vector store created: {store_name} ({vector_store.id})")
            return details
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return {}
    
    def upload_file(self, file_path: str, vector_store_id: str) -> Dict[str, Any]:
        """Upload a file to the vector store for analysis."""
        file_name = os.path.basename(file_path)
        try:
            file_response = client.files.create(file=open(file_path, 'rb'), purpose="assistants")
            attach_response = client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_response.id
            )
            status = {"file": file_name, "status": "success", "file_id": file_response.id}
            self.uploaded_files[file_name] = status
            logger.info(f"File uploaded: {file_name}")
            return status
        except Exception as e:
            logger.error(f"Error uploading {file_name}: {str(e)}")
            return {"file": file_name, "status": "failed", "error": str(e)}
    
    def list_vector_stores(self) -> List[Dict[str, Any]]:
        """List all available vector stores."""
        return list(self.vector_stores.values())
    
    def list_files(self) -> List[Dict[str, Any]]:
        """List all uploaded files."""
        return list(self.uploaded_files.values())

# Function Tools for Document Analysis
@function_tool
def extract_entities(text: str, entity_types: List[str] = ["PERSON", "ORG", "DATE", "GPE"]) -> Dict[str, List[str]]:
    """
    Extract named entities from text.
    
    Args:
        text: The text to analyze
        entity_types: Types of entities to extract (PERSON, ORG, DATE, GPE, etc.)
        
    Returns:
        Dictionary mapping entity types to lists of extracted entities
    """
    # Simplified implementation (in production, use spaCy or similar)
    result = {entity_type: [] for entity_type in entity_types}
    
    # Simple pattern matching for demonstration
    if "PERSON" in entity_types:
        # Look for capitalized words that might be names
        potential_names = re.findall(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+)\b', text)
        result["PERSON"] = list(set(potential_names))
    
    if "DATE" in entity_types:
        # Simple date patterns
        dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text)
        dates += re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text)
        result["DATE"] = list(set(dates))
    
    if "ORG" in entity_types:
        # Look for capitalized words that might be organizations
        orgs = re.findall(r'\b[A-Z][A-Z]+\b', text)  # Acronyms
        orgs += re.findall(r'\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Co)\b', text)
        result["ORG"] = list(set(orgs))
    
    return result

@function_tool
def summarize_text(text: str, max_length: int = 200) -> str:
    """
    Generate a concise summary of the provided text.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary in characters
        
    Returns:
        A summarized version of the input text
    """
    # In a real implementation, we would use OpenAI or another ML model
    # This is a simplified version that extracts key sentences
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 3:
        return text  # Text is already short enough
    
    # Find sentences with keywords (simple approach)
    important_words = ["important", "key", "significant", "main", "critical", "essential"]
    scored_sentences = []
    
    for i, sentence in enumerate(sentences):
        score = 0
        # Score based on position (first and last sentences often contain key info)
        if i == 0 or i == len(sentences) - 1:
            score += 2
            
        # Score based on length (avoid very short sentences)
        if len(sentence.split()) > 5:
            score += 1
            
        # Score based on important words
        for word in important_words:
            if word.lower() in sentence.lower():
                score += 2
                
        scored_sentences.append((score, sentence))
    
    # Sort by score
    scored_sentences.sort(reverse=True)
    
    # Build summary from top sentences
    summary = ""
    for _, sentence in scored_sentences[:3]:
        summary += sentence + " "
        
        if len(summary) >= max_length:
            break
    
    return summary.strip()

@function_tool
def analyze_sentiment(text: str) -> Dict[str, Union[str, float]]:
    """
    Analyze the sentiment of the provided text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Sentiment analysis results including sentiment label and score
    """
    # Simple keyword-based sentiment analysis (for demonstration)
    positive_words = ["good", "great", "excellent", "positive", "amazing", "wonderful", 
                       "fantastic", "outstanding", "superb", "happy", "pleased"]
    negative_words = ["bad", "terrible", "awful", "negative", "horrible", "disappointing", 
                       "poor", "unacceptable", "inadequate", "sad", "angry"]
    
    text_lower = text.lower()
    positive_count = sum(text_lower.count(word) for word in positive_words)
    negative_count = sum(text_lower.count(word) for word in negative_words)
    
    total_count = positive_count + negative_count
    if total_count == 0:
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "explanation": "No strong sentiment indicators found."
        }
    
    score = (positive_count - negative_count) / (positive_count + negative_count)
    
    if score > 0.25:
        sentiment = "positive"
        explanation = f"Text contains {positive_count} positive indicators vs {negative_count} negative."
    elif score < -0.25:
        sentiment = "negative"
        explanation = f"Text contains {negative_count} negative indicators vs {positive_count} positive."
    else:
        sentiment = "neutral"
        explanation = f"Text contains a balanced mix of {positive_count} positive and {negative_count} negative indicators."
    
    return {
        "sentiment": sentiment,
        "score": round(score, 2),
        "explanation": explanation
    }

@function_tool
def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extract key phrases from the provided text.
    
    Args:
        text: The text to analyze
        max_phrases: Maximum number of key phrases to return
        
    Returns:
        List of key phrases
    """
    # Simple implementation based on frequency (in production use TextRank or similar)
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords (simplified list)
    stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being']
    
    words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Find bigrams (pairs of consecutive words)
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append(words[i] + " " + words[i+1])
    
    # Count bigram frequencies
    bigram_freq = {}
    for bigram in bigrams:
        bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
    
    # Combine top words and bigrams
    candidates = [(word, freq) for word, freq in word_freq.items()]
    candidates += [(bigram, freq * 1.5) for bigram, freq in bigram_freq.items()]  # Weight bigrams higher
    
    # Sort by frequency
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Return top phrases
    return [candidate[0] for candidate in candidates[:max_phrases]]

@function_tool
def detect_language(text: str) -> Dict[str, Any]:
    """
    Detect the language of the provided text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with detected language and confidence score
    """
    # Simple language detection based on character frequency
    # In production, use a proper library like langdetect
    
    # Common letter frequencies in various languages
    language_profiles = {
        "english": {'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0, 'n': 6.7, 's': 6.3},
        "spanish": {'e': 13.7, 'a': 12.5, 'o': 8.7, 's': 7.9, 'n': 7.0, 'r': 6.9, 'i': 6.3},
        "french": {'e': 14.7, 'a': 8.0, 's': 7.9, 'i': 7.3, 't': 7.2, 'n': 7.1, 'r': 6.6},
        "german": {'e': 16.4, 'n': 9.8, 'i': 7.6, 's': 7.3, 'r': 7.0, 't': 6.1, 'a': 6.5},
    }
    
    # Count letter frequencies in input text
    text_lower = text.lower()
    letter_count = {}
    total_letters = 0
    
    for char in text_lower:
        if char.isalpha():
            letter_count[char] = letter_count.get(char, 0) + 1
            total_letters += 1
    
    if total_letters == 0:
        return {"language": "unknown", "confidence": 0.0}
    
    # Calculate frequency percentages
    for char in letter_count:
        letter_count[char] = (letter_count[char] / total_letters) * 100
    
    # Compare with language profiles
    language_scores = {}
    for language, profile in language_profiles.items():
        score = 0
        for char, expected_freq in profile.items():
            actual_freq = letter_count.get(char, 0)
            # Calculate similarity (lower difference is better)
            score += 1 / (1 + abs(actual_freq - expected_freq))
        
        language_scores[language] = score
    
    # Find best match
    best_language = max(language_scores.items(), key=lambda x: x[1])
    
    # Calculate confidence (normalized score)
    total_score = sum(language_scores.values())
    confidence = best_language[1] / total_score if total_score > 0 else 0
    
    return {
        "language": best_language[0],
        "confidence": round(confidence, 2),
        "details": {lang: round(score/total_score, 2) for lang, score in language_scores.items()}
    }

@function_tool
def extract_tables_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract tabular data from text content.
    
    Args:
        text: The text containing potential tables
        
    Returns:
        List of extracted tables as dictionaries
    """
    # Look for text that resembles tables
    # This is a simplified implementation; in production use more sophisticated parsing
    
    tables = []
    
    # Find table-like structures using regex
    # This simple approach looks for lines with multiple pipe characters or multiple spaces
    table_candidates = re.findall(r'((?:[^\n]*\|[^\n]*\|[^\n]*\n){2,})', text)
    table_candidates += re.findall(r'((?:[^\n]*  +[^\n]*  +[^\n]*\n){2,})', text)
    
    for idx, table_text in enumerate(table_candidates):
        lines = table_text.strip().split('\n')
        
        # Determine delimiter
        delimiter = '|' if '|' in lines[0] else None
        
        if delimiter:
            # Process pipe-delimited table
            headers = [cell.strip() for cell in lines[0].split(delimiter) if cell.strip()]
            rows = []
            
            for line in lines[1:]:
                if not line.strip():
                    continue
                cells = [cell.strip() for cell in line.split(delimiter)]
                if len(cells) >= len(headers):
                    row_data = {headers[i]: cells[i] for i in range(len(headers))}
                    rows.append(row_data)
            
            tables.append({
                "table_id": idx + 1,
                "headers": headers,
                "rows": rows,
                "row_count": len(rows)
            })
        else:
            # Try to process space-delimited table
            # This is more complex and error-prone
            # For simplicity, we'll just detect potential columns by consistent spacing
            
            # Find column boundaries by looking at spaces
            potential_columns = []
            for line in lines[:3]:  # Look at first few lines to determine structure
                last_pos = 0
                for match in re.finditer(r'  +', line):
                    start, end = match.span()
                    if start > last_pos:
                        potential_columns.append((last_pos, start))
                    last_pos = end
                potential_columns.append((last_pos, len(line)))
            
            # Use the most common column boundaries
            if potential_columns:
                # Simplified: just use first line's boundaries
                columns = potential_columns[:len(potential_columns)//3]
                
                # Extract data using these column boundaries
                headers = []
                for start, end in columns:
                    if start < len(lines[0]):
                        header = lines[0][start:end].strip()
                        headers.append(header)
                
                rows = []
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    
                    row_data = {}
                    for i, (start, end) in enumerate(columns):
                        if i < len(headers) and start < len(line):
                            cell_value = line[start:min(end, len(line))].strip()
                            row_data[headers[i]] = cell_value
                    
                    if row_data:
                        rows.append(row_data)
                
                tables.append({
                    "table_id": idx + 1,
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows)
                })
    
    return tables

@function_tool
def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file metadata
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    try:
        file_stat = os.stat(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "file_size_bytes": file_stat.st_size,
            "file_size_formatted": format_file_size(file_stat.st_size),
            "created_time": datetime.datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            "modified_time": datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "file_extension": file_ext,
            "mime_type": guess_mime_type(file_ext)
        }
        
        return metadata
    
    except Exception as e:
        return {"error": f"Error getting metadata: {str(e)}"}

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def guess_mime_type(extension: str) -> str:
    """Guess MIME type from file extension."""
    mime_types = {
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".csv": "text/csv",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".json": "application/json",
        ".xml": "application/xml",
        ".html": "text/html",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp3": "audio/mpeg",
        ".mp4": "video/mp4"
    }
    return mime_types.get(extension, "application/octet-stream")

@function_tool
def search_text_in_files(search_query: str, vector_store_ids: List[str]) -> Dict[str, Any]:
    """
    Search for text across documents in the vector stores.
    
    Args:
        search_query: The search query
        vector_store_ids: List of vector store IDs to search in
        
    Returns:
        Search results with matching passages
    """
    try:
        results = []
        for vector_store_id in vector_store_ids:
            response = client.vector_stores.query(
                vector_store_id=vector_store_id,
                query=search_query,
                max_passages=3
            )
            
            for passage in response.passages:
                results.append({
                    "file_name": passage.metadata.get("file_name", "Unknown file"),
                    "text": passage.text,
                    "score": passage.score,
                    "vector_store_id": vector_store_id
                })
        
        # Sort by score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return {
            "query": search_query,
            "total_results": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error searching text: {str(e)}")
        return {
            "query": search_query,
            "error": str(e),
            "results": []
        }

# Voice Utilities
class VoiceManager:
    """Manages voice recording, playback, and Text-to-Speech operations."""
    
    def __init__(self):
        self.samplerate = sd.query_devices(kind='input')['default_samplerate']
        self.tts_service = TextToSpeechService()
    
    async def record_audio(self) -> np.ndarray:
        """Record audio from the microphone until user stops."""
        print("Listening... Press Enter to stop.")
        recorded_chunks = []
        
        # Start recording
        with sd.InputStream(samplerate=self.samplerate, channels=1, dtype='int16', 
                           callback=lambda indata, frames, time, status: recorded_chunks.append(indata.copy())):
            input()  # Wait for Enter key to stop recording
        
        # Concatenate chunks
        if recorded_chunks:
            recording = np.concatenate(recorded_chunks, axis=0)
            return recording
        
        return np.array([], dtype=np.int16)
    
    def play_audio(self, audio_data: np.ndarray):
        """Play audio data through speakers."""
        if len(audio_data) > 0:
            sd.play(audio_data, self.samplerate)
            sd.wait()
    
    async def text_to_speech(self, text: str) -> np.ndarray:
        """Convert text to speech using the TTS service."""
        try:
            audio_data = await self.tts_service.synthesize(text)
            return audio_data
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            # Return empty array if TTS fails
            return np.array([], dtype=np.int16)

# Agent Definitions
# Document Analysis Agent
document_analysis_agent = Agent(
    name="DocumentAnalysisAgent",
    instructions="""
    You are a specialized document analysis agent. When provided with text from documents, you:
    1. Identify key information, entities, and relationships
    2. Extract structured data like tables when present
    3. Provide summaries when requested
    4. Answer specific questions about document content
    
    Use the provided tools to analyze document content thoroughly and accurately.
    Always provide precise, factual responses based solely on the document content.
    """,
    tools=[
        extract_entities, 
        summarize_text, 
        analyze_sentiment, 
        extract_key_phrases,
        detect_language,
        extract_tables_from_text
    ]
)

# File Management Agent
file_management_agent = Agent(
    name="FileManagementAgent",
    instructions="""
    You manage file operations and metadata extraction. Your responsibilities include:
    1. Retrieving file metadata
    2. Searching for specific content across files
    3. Providing information about file formats and structures
    
    Always verify file paths before operations and provide clear error messages when issues occur.
    """,
    tools=[
        get_file_metadata,
        search_text_in_files
    ]
)

# Web Research Agent
web_research_agent = Agent(
    name="WebResearchAgent",
    instructions="""
    You conduct web searches to supplement document analysis with real-time information.
    When the user needs information that might not be in their documents, you:
    1. Perform targeted web searches
    2. Extract relevant information from search results
    3. Provide context that complements document content
    
    Focus on authoritative sources and recent information. Always clearly distinguish
    between information from the user's documents and information from web searches.
    """,
    tools=[WebSearchTool()]
)

# Main Triage Agent
main_agent = Agent(
    name="DocumentAnalysisAssistant",
    instructions=prompt_with_handoff_instructions("""
    You are an advanced document analysis assistant that helps users understand and extract insights from documents.
    
    Welcome the user and ask how you can help with their documents. Based on the user's request:
    
    1. For detailed document content analysis, summaries, entity extraction, or sentiment analysis:
       Route to DocumentAnalysisAgent
    
    2. For file operations, metadata extraction, or searching across multiple files:
       Route to FileManagementAgent
    
    3. For supplementary information that requires real-time web search:
       Route to WebResearchAgent
    
    Always maintain a helpful, professional tone and ask clarifying questions when needed.
    """),
    handoffs=[document_analysis_agent, file_management_agent, web_research_agent]
)

# Main Application Class
class DocumentAnalysisPipeline:
    """Main application class for the document analysis pipeline."""
    
    def __init__(self):
        self.doc_manager = DocumentManager()
        self.voice_manager = VoiceManager()
        self.vector_store_id = None
        self.uploaded_file_paths = []
    
    async def initialize(self):
        """Initialize the pipeline by creating a vector store."""
        print("Initializing Document Analysis Pipeline...")
        store_name = f"DocumentAnalysis_{int(time.time())}"
        store_details = self.doc_manager.create_vector_store(store_name)
        
        if store_details:
            self.vector_store_id = store_details["id"]
            print(f"Vector store created: {store_details['name']} (ID: {self.vector_store_id})")
        else:
            print("Failed to create vector store. Please check your API key and try again.")
            return False
        
        return True
    
    async def upload_files(self, file_paths: List[str]):
        """Upload files to the vector store."""
        if not self.vector_store_id:
            print("Vector store not initialized. Please initialize first.")
            return
        
        print(f"Uploading {len(file_paths)} files...")
        for file_path in file_paths:
            if os.path.exists(file_path):
                result = self.doc_manager.upload_file(file_path, self.vector_store_id)
                if result["status"] == "success":
                    self.uploaded_file_paths.append(file_path)
                    print(f"Successfully uploaded: {file_path}")
                else:
                    print(f"Failed to upload: {file_path} - {result.get('error', 'Unknown error')}")
            else:
                print(f"File not found: {file_path}")
        
        print(f"Uploaded {len(self.uploaded_file_paths)} files successfully.")
    
    async def process_text_query(self, query: str):
        """Process a text query through the agent pipeline."""
        print(f"\nProcessing query: {query}")
        result = await Runner.run(main_agent, query)
        print("\nResponse:")
        print(result.final_output)
        return result.final_output
    
    async def process_voice_query(self):
        """Process a voice query through the agent pipeline."""
        print("Recording voice query...")
        audio_data = await self.voice_manager.record_audio()
        
        if len(audio_data) == 0:
            print("No audio recorded.")
            return
        
        # Setup voice pipeline
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(main_agent))
        audio_input = AudioInput(buffer=audio_data)
        
        print("Processing voice query...")
        with trace("Document Analysis Voice Assistant"):
            result = await pipeline.run(audio_input)
        
        # Play response
        print("Assistant is responding...")
        response_chunks = []
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                response_chunks.append(event.data)
        
        if response_chunks:
            response_audio = np.concatenate(response_chunks, axis=0)
            self.voice_manager.play_audio(response_audio)
        else:
            print("No audio response generated.")
        
        print("Voice query processing complete.")
    
    async def run_interactive_session(self):
        """Run an interactive session with text and voice queries."""
        if not await self.initialize():
            return
        
        # Get file paths from user
        file_paths_input = input("Enter file paths (comma-separated): ")
        file_paths = [path.strip() for path in file_paths_input.split(",")]
        
        await self.upload_files(file_paths)
        
        print("\n=== Document Analysis Pipeline Ready ===")
        print("Commands:")
        print("  'text' - Enter a text query")
        print("  'voice' - Start a voice query")
        print("  'exit' - Exit the application")
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == "exit":
                print("Exiting application.")
                break
            elif command == "text":
                query = input("Enter your query: ")
                await self.process_text_query(query)
            elif command == "voice":
                await self.process_voice_query()
            else:
                print(f"Unknown command: {command}")
                print("Available commands: text, voice, exit")

# Entry point
async def main():
    pipeline = DocumentAnalysisPipeline()
    await pipeline.run_interactive_session()

if __name__ == "__main__":
    asyncio.run(main())