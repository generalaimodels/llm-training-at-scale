import os
import re
import asyncio
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents import Runner, trace

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"))
set_default_openai_key(os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"))

# --- Document Processing Functions ---

def create_vector_store(store_name: str) -> Dict[str, Any]:
    """Create a vector store for document embeddings."""
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print(f"Vector store created: {store_name} with ID {vector_store.id}")
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}

def upload_file(file_path: str, vector_store_id: str) -> Dict[str, Any]:
    """Upload a file to the vector store for indexing."""
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(file=open(file_path, 'rb'), purpose="assistants")
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        print(f"File {file_name} uploaded successfully to vector store")
        return {"file": file_name, "status": "success", "file_id": file_response.id}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}

def process_documents(file_paths: List[str]) -> Dict[str, Any]:
    """Process multiple documents and create a vector store."""
    store_name = f"Document_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vector_store = create_vector_store(store_name)
    
    if not vector_store:
        return {"status": "failed", "error": "Failed to create vector store"}
    
    results = []
    for file_path in file_paths:
        result = upload_file(file_path, vector_store["id"])
        results.append(result)
    
    return {
        "status": "success",
        "vector_store_id": vector_store["id"],
        "vector_store_name": vector_store["name"],
        "file_results": results
    }

def read_file_content(file_path: str) -> Tuple[str, Optional[str]]:
    """Read the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read(), None
    except UnicodeDecodeError:
        try:
            # Try binary mode for PDFs, etc.
            with open(file_path, 'rb') as f:
                return f"Binary file detected: {os.path.basename(file_path)}", None
        except Exception as e:
            return "", f"Error reading file: {str(e)}"
    except Exception as e:
        return "", f"Error reading file: {str(e)}"

# --- Function Tools for Document Analysis ---

@function_tool
def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text."""
    entities = {
        "people": re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+', text),
        "organizations": re.findall(r'[A-Z][A-Z]+', text),
        "dates": re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}', text),
        "emails": re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text),
    }
    return entities

@function_tool
def summarize_text(text: str, max_length: int = 200) -> str:
    """Generate a concise summary of the text."""
    words = text.split()
    if len(words) <= max_length:
        return text
    
    # Simple extractive summarization
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 3:
        return text
    
    # Take first and last sentence, plus a middle one
    summary = sentences[0] + " " + sentences[len(sentences)//2] + " " + sentences[-1]
    return summary

@function_tool
def compare_documents(doc1_text: str, doc2_text: str) -> Dict[str, Any]:
    """Compare two documents and identify similarities and differences."""
    doc1_words = set(doc1_text.lower().split())
    doc2_words = set(doc2_text.lower().split())
    
    common_words = doc1_words.intersection(doc2_words)
    unique_to_doc1 = doc1_words - doc2_words
    unique_to_doc2 = doc2_words - doc1_words
    
    similarity_score = len(common_words) / (len(doc1_words.union(doc2_words)) or 1)
    
    return {
        "similarity_score": similarity_score,
        "common_word_count": len(common_words),
        "unique_to_doc1_count": len(unique_to_doc1),
        "unique_to_doc2_count": len(unique_to_doc2),
        "sample_common_words": list(common_words)[:10] if common_words else []
    }

@function_tool
def extract_tables(text: str) -> List[Dict[str, Any]]:
    """Extract tabular data from document text."""
    tables = []
    # Look for patterns that might indicate tables
    table_patterns = re.findall(r'(\|.*\|[\r\n]+)+', text)
    
    for i, pattern in enumerate(table_patterns):
        rows = pattern.strip().split('\n')
        headers = [cell.strip() for cell in rows[0].strip('|').split('|')]
        data = []
        
        for row in rows[1:]:
            if row.strip():
                cells = [cell.strip() for cell in row.strip('|').split('|')]
                if len(cells) == len(headers):
                    data.append(dict(zip(headers, cells)))
        
        tables.append({
            "table_id": i + 1,
            "headers": headers,
            "row_count": len(data),
            "data": data
        })
    
    return tables

@function_tool
def sentiment_analysis(text: str) -> Dict[str, Any]:
    """Analyze sentiment of the text."""
    positive_words = ['good', 'great', 'excellent', 'positive', 'happy', 'best', 'love']
    negative_words = ['bad', 'poor', 'negative', 'terrible', 'worst', 'hate', 'awful']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        sentiment = "neutral"
        score = 0
    else:
        score = (pos_count - neg_count) / total
        if score > 0.25:
            sentiment = "positive"
        elif score < -0.25:
            sentiment = "negative"
        else:
            sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "score": score,
        "positive_word_count": pos_count,
        "negative_word_count": neg_count
    }

@function_tool
def extract_keywords(text: str, top_n: int = 10) -> List[Dict[str, Union[str, int]]]:
    """Extract key terms from document text."""
    words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
    stopwords = ['the', 'and', 'to', 'of', 'in', 'a', 'is', 'that', 'for', 'on', 'with']
    
    word_counts = {}
    for word in words:
        if word not in stopwords:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"keyword": k, "count": v} for k, v in keywords]

@function_tool
def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract and return file metadata."""
    try:
        path = Path(file_path)
        stats = path.stat()
        return {
            "filename": path.name,
            "extension": path.suffix,
            "size_bytes": stats.st_size,
            "size_formatted": f"{stats.st_size / 1024:.2f} KB",
            "created_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "is_directory": path.is_dir(),
        }
    except Exception as e:
        return {"error": str(e)}

@function_tool
def search_documents(query: str, vector_store_id: str) -> List[Dict[str, Any]]:
    """Search through documents using vector search."""
    try:
        search_results = client.vector_stores.vector_store_queries.create(
            vector_store_id=vector_store_id,
            query=query,
            max_num_results=5
        )
        
        results = []
        for item in search_results.data:
            results.append({
                "file_id": item.file_id,
                "text": item.text,
                "score": item.score
            })
        
        return results
    except Exception as e:
        return [{"error": str(e)}]

@function_tool
def document_statistics(text: str) -> Dict[str, Any]:
    """Generate statistics about the document text."""
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    paragraphs = re.split(r'\n\s*\n', text)
    
    unique_words = set(word.lower() for word in words)
    
    total_word_length = sum(len(word) for word in words)
    avg_word_length = total_word_length / len(words) if words else 0
    
    sentence_word_counts = [len(re.findall(r'\b\w+\b', s)) for s in sentences if s.strip()]
    avg_sentence_length = sum(sentence_word_counts) / len(sentence_word_counts) if sentence_word_counts else 0
    
    return {
        "word_count": len(words),
        "unique_word_count": len(unique_words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "average_word_length": avg_word_length,
        "average_sentence_length": avg_sentence_length,
        "longest_word": max(words, key=len) if words else None,
        "shortest_word": min(words, key=len) if words else None,
    }

# --- Specialized Agents ---

# Document Analysis Agent
document_agent = Agent(
    name="DocumentAnalysisAgent",
    instructions=(
        "You are specialized in analyzing document content. You can extract information, "
        "perform basic NLP tasks, and answer questions about documents. Use the provided "
        "tools to analyze documents and provide insights. Be thorough in your analysis "
        "and explain your findings clearly."
    ),
    tools=[
        extract_entities, 
        summarize_text, 
        extract_tables, 
        sentiment_analysis, 
        extract_keywords,
        extract_metadata,
        compare_documents,
        document_statistics
    ],
)

# Document Search Agent
document_search_agent = Agent(
    name="DocumentSearchAgent",
    instructions=(
        "You search through indexed documents to find relevant information for the user's query. "
        "Return exact passages from documents that best answer the user's questions. "
        "Be precise and comprehensive in your searches."
    ),
    tools=[FileSearchTool],  # Will be configured at runtime with specific vector_store_id
)

# Web Research Agent
web_research_agent = Agent(
    name="WebResearchAgent",
    instructions=(
        "You perform web searches to supplement document analysis with up-to-date information. "
        "Use the WebSearchTool to find relevant information online. Focus on reputable sources "
        "and verify information when possible."
    ),
    tools=[WebSearchTool()],
)

# --- Main Triage Agent ---
triage_agent = Agent(
    name="DocumentAssistant",
    instructions=prompt_with_handoff_instructions("""
You are an advanced document analysis assistant. Your primary task is to help users analyze and extract insights from documents.
Listen to the user's query carefully to determine what they need:

1. For questions about the content of uploaded documents, use the DocumentSearchAgent.
2. For requests to analyze documents (extract entities, summarize, etc.), use the DocumentAnalysisAgent.
3. If the user needs additional information from the web to complement their analysis, use the WebResearchAgent.

Always ask clarifying questions if the user's query is ambiguous. Be helpful, detailed, and accurate in your responses.
Provide concise but comprehensive answers. If a user's query is complex, break it down into manageable parts.
"""),
    handoffs=[document_search_agent, document_agent, web_research_agent],
)

class DocumentAnalysisPipeline:
    """Main class to manage the document analysis pipeline."""
    
    def __init__(self):
        """Initialize the document analysis pipeline."""
        self.vector_store_id = None
        self.vector_store_name = None
        self.uploaded_files = []
        self.file_paths = []
        self.file_contents = {}
    
    def initialize_agents(self) -> None:
        """Configure agents with the current vector store."""
        global document_search_agent
        
        if self.vector_store_id:
            # Create a new FileSearchTool with the current vector_store_id
            file_search_tool = FileSearchTool(
                max_num_results=5,
                vector_store_ids=[self.vector_store_id],
            )
            
            # Update the document_search_agent with the new tool
            document_search_agent = Agent(
                name="DocumentSearchAgent",
                instructions=(
                    "You search through indexed documents to find relevant information for the user's query. "
                    "Return exact passages from documents that best answer the user's questions. "
                    "Be precise and comprehensive in your searches."
                ),
                tools=[file_search_tool],
            )
    
    def process_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process the uploaded files and create a vector store."""
        self.file_paths = file_paths
        result = process_documents(file_paths)
        
        if result["status"] == "success":
            self.vector_store_id = result["vector_store_id"]
            self.vector_store_name = result["vector_store_name"]
            self.uploaded_files = [r["file"] for r in result["file_results"] if r["status"] == "success"]
            self.initialize_agents()
            
            # Load file contents for direct access
            for file_path in file_paths:
                content, error = read_file_content(file_path)
                if not error:
                    self.file_contents[os.path.basename(file_path)] = content
        
        return result
    
    async def process_query(self, query: str) -> str:
        """Process a user query using the appropriate agent."""
        if not self.vector_store_id:
            return "Please upload documents first."
        
        try:
            with trace(f"Document Query: {query[:30]}..."):
                result = await Runner.run(triage_agent, query)
            return result.final_output
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def get_file_content(self, filename: str) -> str:
        """Get the content of a specific file."""
        if filename in self.file_contents:
            return self.file_contents[filename]
        
        # Try to find the file in the uploaded files
        for file_path in self.file_paths:
            if os.path.basename(file_path) == filename:
                content, error = read_file_content(file_path)
                if error:
                    return f"Error: {error}"
                self.file_contents[filename] = content
                return content
        
        return f"Error: File '{filename}' not found"
    
    def get_file_list(self) -> List[str]:
        """Get the list of uploaded files."""
        return self.uploaded_files
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the pipeline."""
        return {
            "vector_store_id": self.vector_store_id,
            "vector_store_name": self.vector_store_name,
            "uploaded_files": self.uploaded_files,
            "file_count": len(self.uploaded_files)
        }

async def main():
    """Main function to run the document analysis pipeline."""
    print("\n=================================================")
    print("   Advanced Document Analysis Pipeline v1.0")
    print("=================================================\n")
    
    pipeline = DocumentAnalysisPipeline()
    
    # Request file paths from user
    print("Please enter the paths to 2-3 files for analysis (one per line):")
    print("When finished, enter a blank line.")
    
    file_paths = []
    while True:
        file_path = input("File path: ").strip()
        if not file_path:
            if file_paths:  # Only break if we have at least one file
                break
            else:
                print("Please enter at least one file path.")
                continue
        
        if os.path.exists(file_path):
            file_paths.append(file_path)
            print(f"Added: {file_path}")
        else:
            print(f"File not found: {file_path}")
    
    # Process the files
    print("\nProcessing files...")
    result = pipeline.process_files(file_paths)
    
    if result["status"] != "success":
        print(f"Error processing files: {result.get('error', 'Unknown error')}")
        return
    
    print(f"\nSuccessfully processed {len(pipeline.uploaded_files)} files:")
    for file in pipeline.uploaded_files:
        print(f"- {file}")
    
    print(f"\nVector store created: {pipeline.vector_store_name}")
    
    # Command-line interface for queries
    print("\n=================================================")
    print("           Document Analysis Interface")
    print("=================================================")
    print("\nAvailable commands:")
    print("  /help               - Show this help message")
    print("  /files              - List uploaded files")
    print("  /status             - Show pipeline status")
    print("  /exit               - Exit the program")
    print("  any other text      - Process as a query")
    
    while True:
        query = input("\nQuery: ").strip()
        
        if not query:
            continue
        
        if query.lower() == '/exit':
            break
        elif query.lower() == '/help':
            print("\nAvailable commands:")
            print("  /help               - Show this help message")
            print("  /files              - List uploaded files")
            print("  /status             - Show pipeline status")
            print("  /exit               - Exit the program")
            print("  any other text      - Process as a query")
        elif query.lower() == '/files':
            files = pipeline.get_file_list()
            print("\nUploaded files:")
            for i, file in enumerate(files, 1):
                print(f"  {i}. {file}")
        elif query.lower() == '/status':
            status = pipeline.get_status()
            print("\nPipeline status:")
            print(f"  Vector store: {status['vector_store_name']}")
            print(f"  Files: {status['file_count']}")
        else:
            print("\nProcessing query...")
            response = await pipeline.process_query(query)
            print("\nResponse:")
            print(response)
    
    print("\nThank you for using the Advanced Document Analysis Pipeline!")

if __name__ == "__main__":
    asyncio.run(main())