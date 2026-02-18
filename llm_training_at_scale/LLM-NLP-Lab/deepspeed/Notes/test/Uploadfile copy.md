**Uploads API Documentation**
=====================================

**Overview**
------------

The `Uploads` API provides a set of methods for uploading files to the OpenAI platform. It supports both synchronous and asynchronous uploads, allowing you to handle large files by splitting them into smaller parts.

**Main API Classes**
--------------------

*   **[`Uploads`](#uploads-class)**: Synchronous uploads API.
*   **[`AsyncUploads`](#asyncuploads-class)**: Asynchronous uploads API.

**Constants**
-------------

*   **`DEFAULT_PART_SIZE`**: The default size of each file part in bytes (64MB).

    ```python
DEFAULT_PART_SIZE = 64 * 1024 * 1024  # 64MB
```

**Uploads Class**
-----------------

### **`Uploads`**

```python
class Uploads(SyncAPIResource)
```

The `Uploads` class provides synchronous methods for uploading files.

#### **Properties**

*   **`parts`** (`Parts`): A cached property that returns a `Parts` object for managing file parts.

    ```python
@cached_property
def parts(self) -> Parts:
    return Parts(self._client)
```

*   **`with_raw_response`** (`UploadsWithRawResponse`): Returns the raw response object instead of parsed content for any HTTP method call.

    ```python
@cached_property
def with_raw_response(self) -> UploadsWithRawResponse:
    """
    For more information, see https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers
    """
    return UploadsWithRawResponse(self)
```

*   **`with_streaming_response`** (`UploadsWithStreamingResponse`): An alternative to `.with_raw_response` that doesn't eagerly read the response body.

    ```python
@cached_property
def with_streaming_response(self) -> UploadsWithStreamingResponse:
    """
    For more information, see https://www.github.com/openai/openai-python#with_streaming_response
    """
    return UploadsWithStreamingResponse(self)
```

#### **Methods**

##### **`upload_file_chunked`**

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

Splits the given file into multiple parts and uploads them sequentially.

**Parameters:**

*   **`file`** (`os.PathLike[str] | bytes`): The file path or in-memory bytes.
*   **`mime_type`** (`str`): The MIME type of the file (e.g., "application/pdf").
*   **`purpose`** (`FilePurpose`): The intended purpose of the uploaded file (e.g., "assistants").
*   **`filename`** (`str | None`): The name of the file (required for in-memory files).
*   **`bytes`** (`int | None`): The total size of the file in bytes (required for in-memory files).
*   **`part_size`** (`int | None`): The size of each part (defaults to `DEFAULT_PART_SIZE`).
*   **`md5`** (`str | NotGiven`): The optional MD5 checksum for verifying file integrity.

**Returns:** `Upload` object.

**Raises:**

*   `TypeError`: If `filename` or `bytes` is missing for in-memory files.

**Example:**

```python
from pathlib import Path

upload = client.uploads.upload_file_chunked(
    file=Path("example.pdf"),
    mime_type="application/pdf",
    purpose="assistants",
)
```

**Overloads:**

1.  For file paths:

    ```python
@overload
def upload_file_chunked(
    self,
    *,
    file: os.PathLike[str],
    mime_type: str,
    purpose: FilePurpose,
    bytes: int | None = None,
    part_size: int | None = None,
    md5: str | NotGiven = NOT_GIVEN,
) -> Upload:
```

2.  For in-memory bytes:

    ```python
@overload
def upload_file_chunked(
    self,
    *,
    file: bytes,
    filename: str,
    bytes: int,
    mime_type: str,
    purpose: FilePurpose,
    part_size: int | None = None,
    md5: str | NotGiven = NOT_GIVEN,
) -> Upload:
```

##### **`create`**

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

Creates an intermediate `Upload` object to which parts can be added.

**Parameters:**

*   **`bytes`** (`int`): The total size of the file in bytes.
*   **`filename`** (`str`): The name of the file.
*   **`mime_type`** (`str`): The MIME type of the file.
*   **`purpose`** (`FilePurpose`): The intended purpose of the file.
*   **`extra_headers`**, **`extra_query`**, **`extra_body`**: Optional parameters for customizing the API request.
*   **`timeout`**: Request timeout in seconds.

**Returns:** `Upload` object.

**Raises:**

*   `ValueError`: If required parameters are invalid.

**Example:**

```python
upload = client.uploads.create(
    bytes=1024,
    filename="example.txt",
    mime_type="text/plain",
    purpose="assistants",
)
```

##### **`cancel`**

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

Cancels an ongoing upload. No further parts can be added after cancellation.

**Parameters:**

*   **`upload_id`** (`str`): The ID of the upload to cancel.
*   **`extra_headers`**, **`extra_query`**, **`extra_body`**: Optional parameters for customizing the API request.
*   **`timeout`**: Request timeout in seconds.

**Returns:** `Upload` object.

**Raises:**

*   `ValueError`: If `upload_id` is empty.

**Example:**

```python
upload = client.uploads.cancel(upload_id="upload_123")
```

##### **`complete`**

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

Completes an upload by assembling the parts into a final file.

**Parameters:**

*   **`upload_id`** (`str`): The ID of the upload to complete.
*   **`part_ids`** (`List[str]`): Ordered list of part IDs.
*   **`md5`** (`str | NotGiven`): Optional MD5 checksum for verification.
*   **`extra_headers`**, **`extra_query`**, **`extra_body`**: Optional parameters for customizing the API request.
*   **`timeout`**: Request timeout in seconds.

**Returns:** `Upload` object containing the final `File` object.

**Raises:**

*   `ValueError`: If `upload_id` is empty or `part_ids` is invalid.

**Example:**

```python
upload = client.uploads.complete(
    upload_id="upload_123",
    part_ids=["part_1", "part_2", "part_3"],
    md5="d41d8cd98f00b204e9800998ecf8427e",
)
```

**AsyncUploads Class**
----------------------

### **`AsyncUploads`**

```python
class AsyncUploads(AsyncAPIResource)
```

The `AsyncUploads` class mirrors the `Uploads` API but provides asynchronous methods.

#### **Properties**

*   **`parts`** (`AsyncParts`): A cached property for managing file parts asynchronously.

    ```python
@cached_property
def parts(self) -> AsyncParts:
    return AsyncParts(self._client)
```

*   **`with_raw_response`** (`AsyncUploadsWithRawResponse`): Returns the raw response object for async HTTP calls.

    ```python
@cached_property
def with_raw_response(self) -> AsyncUploadsWithRawResponse:
    """
    For more information, see https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers
    """
    return AsyncUploadsWithRawResponse(self)
```

**Methods**

All methods in `AsyncUploads` are asynchronous versions of those in `Uploads`. Replace `client.uploads` with `await client.uploads` and use `async`/`await` syntax.

**Example:**

```python
async def upload_file():
    upload = await client.uploads.create(
        bytes=1024,
        filename="example.txt",
        mime_type="text/plain",
        purpose="assistants",
    )
    # Continue with async operations
```

**Type References**
--------------------

*   **`FilePurpose`**: Enum defining the purpose of the uploaded file (e.g., "assistants", "vision").
*   **`Upload`**: Object representing an upload session.
*   **`Parts`**: Synchronous API for managing file parts.
*   **`AsyncParts`**: Asynchronous API for managing file parts.

**Error Handling**
------------------

*   **`ValueError`**: Raised for invalid arguments (e.g., empty `upload_id`).
*   **`TypeError`**: Raised for missing required parameters (e.g., `filename` for in-memory files).

**Best Practices**
------------------

1.  Always validate `mime_type` and `purpose` against supported values in the [OpenAI documentation](https://platform.openai.com/docs/api-reference/files/create).
2.  Use `md5` checksums to verify file integrity during uploads.
3.  Handle `Timeout` exceptions for large uploads.

**Links**
----------

*   [OpenAI File API Reference](https://platform.openai.com/docs/api-reference/files/object)
*   [OpenAI Uploads API Reference](https://platform.openai.com/docs/api-reference/uploads/object)
*   [Python Client GitHub](https://www.github.com/openai/openai-python)




Here's a comprehensive, end-to-end example that demonstrates how to use the `Uploads` and `AsyncUploads` classes. This script covers:

1. **Synchronous Uploads**: Creating an upload, uploading file parts, completing the upload, and cancelling an upload.
2. **Asynchronous Uploads**: All the above steps using `AsyncUploads`.
3. **Error Handling**: Includes checks for common exceptions like `ValueError`, `TypeError`, and timeouts.
4. **Raw Response Handling**: Demonstrates how to access raw HTTP responses.
5. **MD5 Verification**: Validates file integrity using MD5 checksums.

**Prerequisites**:

* Install the OpenAI Python client:
  ```bash
pip install openai httpx
```
* Set your OpenAI API key as an environment variable:
  ```bash
export OPENAI_API_KEY="your_api_key_here"
```

**Full Example Code (`uploads_demo.py`):**

```python
import os
import io
import hashlib
import asyncio
from pathlib import Path
from typing import List
import logging

# Mock imports (replace these with actual imports from the OpenAI library)
from openai import OpenAI, AsyncOpenAI
from openai.types import FilePurpose, Upload
from openai._types import NotGiven, NOT_GIVEN
from openai._utils import maybe_transform
from httpx import Headers, Query, Body

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Constants
TEST_FILE_PATH = Path("test_upload.pdf")  # Create a test file or replace this
TEST_FILE_MIME = "application/pdf"
TEST_FILE_PURPOSE = FilePurpose.ASSISTANTS
DEFAULT_PART_SIZE = 64 * 1024 * 1024  # 64MB

# Initialize clients
sync_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            md5.update(chunk)
    return md5.hexdigest()


async def async_calculate_md5(file_path: Path) -> str:
    """Async-friendly MD5 calculation."""
    return calculate_md5(file_path)


def sync_upload_workflow(client):
    """Demonstrates synchronous upload workflow."""
    log.info("=== Synchronous Upload Workflow ===")

    # Step 1: Create an upload session
    log.info("1. Creating upload session...")
    upload = client.uploads.create(
        bytes=TEST_FILE_PATH.stat().st_size,
        filename=TEST_FILE_PATH.name,
        mime_type=TEST_FILE_MIME,
        purpose=TEST_FILE_PURPOSE,
    )
    log.info(f"Upload ID: {upload.id}")

    # Step 2: Upload file parts
    log.info("2. Uploading file parts...")
    part_ids: List[str] = []
    with open(TEST_FILE_PATH, "rb") as f:
        while True:
            data = f.read(DEFAULT_PART_SIZE)
            if not data:
                break
            part = client.uploads.parts.create(upload_id=upload.id, data=data)
            log.info(f"Uploaded part {part.id}")
            part_ids.append(part.id)

    # Step 3: Complete the upload with MD5 verification
    log.info("3. Completing upload...")
    md5_checksum = calculate_md5(TEST_FILE_PATH)
    completed_upload = client.uploads.complete(
        upload_id=upload.id,
        part_ids=part_ids,
        md5=md5_checksum,
    )
    log.info(f"Upload completed: {completed_upload.id}")
    log.info(f"File ID: {completed_upload.file.id}")

    # Step 4: Try cancelling (for demonstration, create a new upload)
    log.info("4. Testing cancellation...")
    cancel_upload = client.uploads.create(
        bytes=1024,
        filename="cancel_test.txt",
        mime_type="text/plain",
        purpose=TEST_FILE_PURPOSE,
    )
    cancelled = client.uploads.cancel(upload_id=cancel_upload.id)
    log.info(f"Cancellation status: {cancelled.status}")

    # Step 5: Raw response example
    log.info("5. Fetching raw response...")
    raw_upload = client.uploads.with_raw_response.create(
        bytes=1024,
        filename="raw_test.txt",
        mime_type="text/plain",
        purpose=TEST_FILE_PURPOSE,
    )
    log.info(f"Raw headers: {raw_upload.headers}")


async def async_upload_workflow(client):
    """Demonstrates asynchronous upload workflow."""
    log.info("=== Asynchronous Upload Workflow ===")

    # Step 1: Create an upload session
    log.info("1. Creating upload session...")
    upload = await client.uploads.create(
        bytes=TEST_FILE_PATH.stat().st_size,
        filename=TEST_FILE_PATH.name,
        mime_type=TEST_FILE_MIME,
        purpose=TEST_FILE_PURPOSE,
    )
    log.info(f"Upload ID: {upload.id}")

    # Step 2: Upload file parts
    log.info("2. Uploading file parts...")
    part_ids: List[str] = []
    with open(TEST_FILE_PATH, "rb") as f:
        while True:
            data = f.read(DEFAULT_PART_SIZE)
            if not data:
                break
            part = await client.uploads.parts.create(upload_id=upload.id, data=data)
            log.info(f"Uploaded part {part.id}")
            part_ids.append(part.id)

    # Step 3: Complete the upload with MD5 verification
    log.info("3. Completing upload...")
    md5_checksum = await async_calculate_md5(TEST_FILE_PATH)
    completed_upload = await client.uploads.complete(
        upload_id=upload.id,
        part_ids=part_ids,
        md5=md5_checksum,
    )
    log.info(f"Upload completed: {completed_upload.id}")
    log.info(f"File ID: {completed_upload.file.id}")

    # Step 4: Try cancelling (for demonstration, create a new upload)
    log.info("4. Testing cancellation...")
    cancel_upload = await client.uploads.create(
        bytes=1024,
        filename="async_cancel_test.txt",
        mime_type="text/plain",
        purpose=TEST_FILE_PURPOSE,
    )
    cancelled = await client.uploads.cancel(upload_id=cancel_upload.id)
    log.info(f"Cancellation status: {cancelled.status}")

    # Step 5: Raw response example
    log.info("5. Fetching raw response...")
    raw_upload = await client.uploads.with_raw_response.create(
        bytes=1024,
        filename="async_raw_test.txt",
        mime_type="text/plain",
        purpose=TEST_FILE_PURPOSE,
    )
    log.info(f"Raw headers: {raw_upload.headers}")


def error_handling_examples(client):
    """Demonstrates error handling."""
    log.info("=== Error Handling Examples ===")

    # Case 1: Missing filename for in-memory upload
    try:
        log.info("Testing missing filename...")
        client.uploads.upload_file_chunked(
            file=b"Hello, world!",
            mime_type="text/plain",
            purpose=TEST_FILE_PURPOSE,
        )
    except TypeError as e:
        log.error(f"Caught error: {e}")

    # Case 2: Invalid upload ID
    try:
        log.info("Testing invalid upload ID...")
        client.uploads.cancel(upload_id="")
    except ValueError as e:
        log.error(f"Caught error: {e}")


async def async_error_handling_examples(client):
    """Demonstrates async error handling."""
    log.info("=== Async Error Handling Examples ===")

    # Case 1: Missing filename for in-memory upload
    try:
        log.info("Testing missing filename...")
        await client.uploads.upload_file_chunked(
            file=b"Hello, world!",
            mime_type="text/plain",
            purpose=TEST_FILE_PURPOSE,
        )
    except TypeError as e:
        log.error(f"Caught error: {e}")

    # Case 2: Invalid upload ID
    try:
        log.info("Testing invalid upload ID...")
        await client.uploads.cancel(upload_id="")
    except ValueError as e:
        log.error(f"Caught error: {e}")


if __name__ == "__main__":
    # Ensure test file exists
    if not TEST_FILE_PATH.exists():
        with open(TEST_FILE_PATH, "w") as f:
            f.write("This is a test file for uploads demo.")
        log.warning(f"Created dummy file: {TEST_FILE_PATH}")

    # Run synchronous workflow
    log.info("\nStarting synchronous workflow...")
    sync_upload_workflow(sync_client)

    # Run error handling examples
    log.info("\nRunning sync error handling tests...")
    error_handling_examples(sync_client)

    # Run asynchronous workflow
    log.info("\nStarting asynchronous workflow...")
    asyncio.run(async_upload_workflow(async_client))

    # Run async error handling examples
    log.info("\nRunning async error handling tests...")
    asyncio.run(async_error_handling_examples(async_client))
```

**How to Run**:

1. Save the script as `uploads_demo.py`.
2. Set your OpenAI API key in the environment.
3. Execute the script:
   ```bash
python uploads_demo.py
```

**Expected Output**:

The script will log each step of the upload process, including:

*   Creation of upload sessions.
*   Part uploads with IDs.
*   Completion with MD5 verification.
*   Cancellation status.
*   Raw HTTP response headers.
*   Error handling cases (missing filename, invalid IDs).

**Notes**:

*   Replace `test_upload.pdf` with an actual file on your system.
*   The script creates a dummy file if none exists.
*   Error cases are intentionally triggered to demonstrate handling.

