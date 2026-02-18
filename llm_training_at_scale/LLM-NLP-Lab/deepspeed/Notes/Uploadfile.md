# Uploads API Documentation

## Constants

```python
DEFAULT_PART_SIZE = 64 * 1024 * 1024  # 64MB
```

## Uploads Class

The `Uploads` class provides synchronous methods for uploading files, particularly handling large files by splitting them into multiple parts.

### Properties

#### `parts`
```python
@cached_property
def parts(self) -> Parts:
```
Returns a `Parts` resource for creating individual file parts.

#### `with_raw_response`
```python
@cached_property
def with_raw_response(self) -> UploadsWithRawResponse:
```
Prefix for any method call to return the raw response object instead of parsed content.

#### `with_streaming_response`
```python
@cached_property
def with_streaming_response(self) -> UploadsWithStreamingResponse:
```
Alternative to `with_raw_response` that doesn't eagerly read the response body.

### Methods

#### `upload_file_chunked`
```python
def upload_file_chunked(
    self,
    *,
    file: os.PathLike[str] | bytes,
    mime_type: str,
    purpose: FilePurpose,
    filename: str | None = None,
    bytes: int | None = None,
    part_size: int | None = None,
    md5: str | NotGiven = NOT_GIVEN,
) -> Upload:
```
Splits a file into multiple parts (default 64MB) and uploads them sequentially.

**Parameters:**
- `file`: File path or bytes content
- `mime_type`: MIME type of the file
- `purpose`: Intended purpose of the uploaded file
- `filename`: Name of the file (required for in-memory files)
- `bytes`: File size in bytes (required for in-memory files)
- `part_size`: Size of each part in bytes (defaults to 64MB)
- `md5`: Optional MD5 checksum

**Returns:** `Upload` object

**Example:**
```python
from pathlib import Path

client.uploads.upload_file_chunked(
    file=Path("my-paper.pdf"),
    mime_type="pdf",
    purpose="assistants",
)
```

#### `create`
```python
def create(
    self,
    *,
    bytes: int,
    filename: str,
    mime_type: str,
    purpose: FilePurpose,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> Upload:
```
Creates an intermediate upload object that can accept parts.

**Parameters:**
- `bytes`: Number of bytes in the file
- `filename`: Name of the file to upload
- `mime_type`: MIME type of the file
- `purpose`: Intended purpose of the uploaded file
- `extra_headers`: Additional headers for the request
- `extra_query`: Additional query parameters
- `extra_body`: Additional JSON properties
- `timeout`: Request timeout override

**Returns:** `Upload` object

#### `cancel`
```python
def cancel(
    self,
    upload_id: str,
    *,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> Upload:
```
Cancels an upload. No parts may be added after cancellation.

**Parameters:**
- `upload_id`: ID of the upload to cancel
- `extra_headers`, `extra_query`, `extra_body`, `timeout`: Request configuration options

**Returns:** `Upload` object

#### `complete`
```python
def complete(
    self,
    upload_id: str,
    *,
    part_ids: List[str],
    md5: str | NotGiven = NOT_GIVEN,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> Upload:
```
Completes the upload, combining all parts into a final file.

**Parameters:**
- `upload_id`: ID of the upload to complete
- `part_ids`: Ordered list of part IDs
- `md5`: Optional MD5 checksum for verification
- `extra_headers`, `extra_query`, `extra_body`, `timeout`: Request configuration options

**Returns:** `Upload` object with a nested `File` object ready for use

## AsyncUploads Class

The `AsyncUploads` class provides asynchronous versions of the same functionality as the `Uploads` class.

### Properties

#### `parts`
```python
@cached_property
def parts(self) -> AsyncParts:
```
Returns an `AsyncParts` resource for creating individual file parts asynchronously.

#### `with_raw_response`
```python
@cached_property
def with_raw_response(self) -> AsyncUploadsWithRawResponse:
```
Prefix for any method call to return the raw response object instead of parsed content.

**Note:** The `AsyncUploads` class contains the same methods as `Uploads` but operates asynchronously.



# Complete End-to-End Example of Uploads API Usage

Below is a comprehensive example showing how to use the Uploads API for both synchronous and asynchronous file uploads:

```python
import os
import asyncio
from pathlib import Path
import hashlib
from openai import OpenAI, AsyncOpenAI

# Configure your API key
api_key = "your-api-key"

# Create clients
sync_client = OpenAI(api_key=api_key)
async_client = AsyncOpenAI(api_key=api_key)

# Example file path
file_path = Path("example_document.pdf")
file_size = file_path.stat().st_size
file_mime_type = "application/pdf"
file_purpose = "assistants"

# Calculate MD5 hash of file (optional but recommended for verification)
def calculate_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

file_md5 = calculate_md5(file_path)

# Example 1: Using the high-level upload_file_chunked method (simplest approach)
def simple_upload():
    print("\n--- Simple Upload Example ---")
    try:
        result = sync_client.uploads.upload_file_chunked(
            file=file_path,
            mime_type=file_mime_type,
            purpose=file_purpose,
            md5=file_md5
        )
        print(f"File uploaded successfully: {result.id}")
        print(f"File ID for future use: {result.file.id}")
        return result
    except Exception as e:
        print(f"Error during simple upload: {e}")

# Example 2: Manual upload process with explicit part creation
def manual_upload():
    print("\n--- Manual Upload Example ---")
    try:
        # Step 1: Create an upload
        upload = sync_client.uploads.create(
            bytes=file_size,
            filename=file_path.name,
            mime_type=file_mime_type,
            purpose=file_purpose
        )
        print(f"Upload created with ID: {upload.id}")
        
        # Step 2: Upload parts
        part_ids = []
        part_size = 64 * 1024 * 1024  # 64MB chunks
        
        with open(file_path, "rb") as f:
            part_number = 1
            while True:
                data = f.read(part_size)
                if not data:
                    break
                    
                part = sync_client.uploads.parts.create(
                    upload_id=upload.id,
                    data=data
                )
                part_ids.append(part.id)
                print(f"Uploaded part {part_number} with ID: {part.id}")
                part_number += 1
        
        # Step 3: Complete the upload
        completed_upload = sync_client.uploads.complete(
            upload_id=upload.id,
            part_ids=part_ids,
            md5=file_md5
        )
        
        print(f"Upload completed successfully: {completed_upload.id}")
        print(f"File ID for future use: {completed_upload.file.id}")
        return completed_upload
    except Exception as e:
        print(f"Error during manual upload: {e}")

# Example 3: Async upload example
async def async_upload():
    print("\n--- Async Upload Example ---")
    try:
        # Step 1: Create an upload
        upload = await async_client.uploads.create(
            bytes=file_size,
            filename=file_path.name,
            mime_type=file_mime_type,
            purpose=file_purpose
        )
        print(f"Async upload created with ID: {upload.id}")
        
        # Step 2: Upload parts
        part_ids = []
        part_size = 64 * 1024 * 1024  # 64MB chunks
        
        with open(file_path, "rb") as f:
            part_number = 1
            while True:
                data = f.read(part_size)
                if not data:
                    break
                    
                part = await async_client.uploads.parts.create(
                    upload_id=upload.id,
                    data=data
                )
                part_ids.append(part.id)
                print(f"Uploaded async part {part_number} with ID: {part.id}")
                part_number += 1
        
        # Step 3: Complete the upload
        completed_upload = await async_client.uploads.complete(
            upload_id=upload.id,
            part_ids=part_ids,
            md5=file_md5
        )
        
        print(f"Async upload completed successfully: {completed_upload.id}")
        print(f"File ID for future use: {completed_upload.file.id}")
        return completed_upload
    except Exception as e:
        print(f"Error during async upload: {e}")

# Example 4: Error handling and cancellation
def upload_with_error_handling():
    print("\n--- Upload with Error Handling Example ---")
    upload_id = None
    try:
        # Step 1: Create an upload
        upload = sync_client.uploads.create(
            bytes=file_size,
            filename=file_path.name,
            mime_type=file_mime_type,
            purpose=file_purpose
        )
        upload_id = upload.id
        print(f"Upload created with ID: {upload.id}")
        
        # Simulate an error during upload
        if True:  # Change to trigger error
            raise ValueError("Simulated error during upload")
            
        # Remaining upload steps would go here
        
    except Exception as e:
        print(f"Error during upload: {e}")
        # Cancel the upload if it was created
        if upload_id:
            try:
                cancelled = sync_client.uploads.cancel(upload_id=upload_id)
                print(f"Upload {upload_id} cancelled")
            except Exception as cancel_err:
                print(f"Error cancelling upload: {cancel_err}")

# Run the examples
if __name__ == "__main__":
    # Run synchronous examples
    simple_upload()
    manual_upload()
    upload_with_error_handling()
    
    # Run async example
    asyncio.run(async_upload())
```

This example demonstrates:

1. **Simple Upload**: Using the high-level `upload_file_chunked` method that handles chunking automatically.
2. **Manual Upload**: The full process of creating an upload, uploading parts, and completing the upload.
3. **Async Upload**: Asynchronous version of the manual upload process.
4. **Error Handling**: How to handle errors and cancel uploads when needed.

The example includes MD5 checksum calculation for file integrity verification and proper handling of chunking for large files. Each part of the process is labeled with print statements so you can follow the execution flow.

