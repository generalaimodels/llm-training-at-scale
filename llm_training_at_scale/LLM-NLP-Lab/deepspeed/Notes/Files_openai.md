# `Files` Class API Documentation

## Class Definition
```python
class Files(SyncAPIResource)
```

The `Files` class provides methods for managing files in the OpenAI platform, including uploading, retrieving, listing, and deleting files.

## Properties

### `with_raw_response`
```python
@cached_property
def with_raw_response(self) -> FilesWithRawResponse
```
Returns a version of the API that returns raw response objects instead of parsed content.

### `with_streaming_response`
```python
@cached_property
def with_streaming_response(self) -> FilesWithStreamingResponse
```
Returns a version of the API that doesn't eagerly read the response body.

## Methods

### `create`
```python
def create(
    self,
    *,
    file: FileTypes,
    purpose: FilePurpose,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> FileObject
```
Uploads a file that can be used across various endpoints.

**Parameters:**
- `file`: The File object to be uploaded
- `purpose`: The intended purpose of the file (`"assistants"`, `"vision"`, `"batch"`, or `"fine-tune"`)
- `extra_headers`: Optional additional headers
- `extra_query`: Optional additional query parameters
- `extra_body`: Optional additional JSON properties
- `timeout`: Override client-level default timeout

**Returns:** `FileObject`

**Notes:**
- Individual files can be up to 512 MB
- Organization limit is 100 GB
- Assistants API supports files up to 2 million tokens
- Fine-tuning API only supports `.jsonl` files
- Batch API supports `.jsonl` files up to 200 MB

**Example:**
```python
file_obj = client.files.create(
    file=open("data.jsonl", "rb"),
    purpose="fine-tune"
)
```

### `retrieve`
```python
def retrieve(
    self,
    file_id: str,
    *,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> FileObject
```
Returns information about a specific file.

**Parameters:**
- `file_id`: ID of the file to retrieve
- `extra_headers`: Optional additional headers
- `extra_query`: Optional additional query parameters
- `extra_body`: Optional additional JSON properties
- `timeout`: Override client-level default timeout

**Returns:** `FileObject`

**Example:**
```python
file_info = client.files.retrieve(file_id="file_abc123")
```

### `list`
```python
def list(
    self,
    *,
    after: str | NotGiven = NOT_GIVEN,
    limit: int | NotGiven = NOT_GIVEN,
    order: Literal["asc", "desc"] | NotGiven = NOT_GIVEN,
    purpose: str | NotGiven = NOT_GIVEN,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> SyncCursorPage[FileObject]
```
Returns a paginated list of files.

**Parameters:**
- `after`: Cursor for pagination (object ID that defines your place in the list)
- `limit`: Maximum number of objects to return (1-10,000, default is 10,000)
- `order`: Sort order by `created_at` timestamp (`"asc"` or `"desc"`)
- `purpose`: Filter to only return files with the given purpose
- `extra_headers`: Optional additional headers
- `extra_query`: Optional additional query parameters
- `extra_body`: Optional additional JSON properties
- `timeout`: Override client-level default timeout

**Returns:** `SyncCursorPage[FileObject]`

**Example:**
```python
# List all files, up to 100 at a time
files = client.files.list(limit=100)

# Get files for a specific purpose sorted by creation date
fine_tune_files = client.files.list(
    purpose="fine-tune",
    order="desc",
    limit=50
)
```

### `delete`
```python
def delete(
    self,
    file_id: str,
    *,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> FileDeleted
```
Deletes a file.

**Parameters:**
- `file_id`: ID of the file to delete
- `extra_headers`: Optional additional headers
- `extra_query`: Optional additional query parameters
- `extra_body`: Optional additional JSON properties
- `timeout`: Override client-level default timeout

**Returns:** `FileDeleted`

**Example:**
```python
response = client.files.delete(file_id="file_abc123")
```

### `content`
```python
def content(
    self,
    file_id: str,
    *,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> _legacy_response.HttpxBinaryResponseContent
```
Returns the contents of the specified file.

**Parameters:**
- `file_id`: ID of the file to retrieve content from
- `extra_headers`: Optional additional headers
- `extra_query`: Optional additional query parameters
- `extra_body`: Optional additional JSON properties
- `timeout`: Override client-level default timeout

**Returns:** `HttpxBinaryResponseContent`

**Example:**
```python
file_content = client.files.content(file_id="file_abc123")
```

### `retrieve_content` (Deprecated)
```python
@typing_extensions.deprecated("The `.content()` method should be used instead")
def retrieve_content(
    self,
    file_id: str,
    ...
)
```
Deprecated method for retrieving file content. Use `content()` instead.

