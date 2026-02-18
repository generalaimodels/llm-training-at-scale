**Fine-Tuning API Documentation**
=====================================

**Overview**
------------

The Fine-Tuning API is a crucial component for customizing and optimizing machine learning models. It provides a programmatic interface for managing fine-tuning jobs, allowing developers to seamlessly integrate model refinement into their applications. This documentation outlines the structure, properties, and usage of the Fine-Tuning API, covering both synchronous (`FineTuning`) and asynchronous (`AsyncFineTuning`) implementations.

**API Classes**
---------------

The Fine-Tuning API is exposed through two primary classes:

1. **`FineTuning` (SyncAPIResource)**: For synchronous operations.
2. **`AsyncFineTuning` (AsyncAPIResource)**: For asynchronous operations.

Both classes share a similar structure but are tailored for different operational paradigms.

**FineTuning (SyncAPIResource)**
-------------------------------

### **Properties**

#### **`jobs`**
```python
@cached_property
def jobs(self) -> Jobs:
    return Jobs(self._client)
```
* **Type:** `Jobs`
* **Description:** Returns a `Jobs` object, which manages fine-tuning jobs. This property is lazily loaded due to `@cached_property`.
* **Usage:** Access and manage fine-tuning jobs synchronously.
* **Example:**
  ```python
  fine_tuning = FineTuning(client)
  jobs = fine_tuning.jobs
  # Now interact with `jobs` for operations like listing, creating, or retrieving jobs.
  ```

#### **`with_raw_response`**
```python
@cached_property
def with_raw_response(self) -> FineTuningWithRawResponse:
    # ...
```
* **Type:** `FineTuningWithRawResponse`
* **Description:** Prefixes HTTP method calls to return the **raw response object** (e.g., headers, status code) instead of parsed content.
* **Use Case:** Inspect response metadata (e.g., rate limits, caching headers).
* **Links:**
  - [Accessing Raw Response Data](https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers)
* **Example:**
  ```python
  raw_ft = fine_tuning.with_raw_response
  response = raw_ft.jobs.list()  # Returns raw HTTP response, not parsed content.
  print(response.headers)  # Access headers, status code, etc.
  ```

#### **`with_streaming_response`**
```python
@cached_property
def with_streaming_response(self) -> FineTuningWithStreamingResponse:
    # ...
```
* **Type:** `FineTuningWithStreamingResponse`
* **Description:** Similar to `with_raw_response` but **does not eagerly read the response body**. Ideal for large responses or streaming use cases.
* **Use Case:** Efficient handling of large datasets or real-time data processing.
* **Links:**
  - [Streaming Responses](https://www.github.com/openai/openai-python#with_streaming_response)
* **Example:**
  ```python
  stream_ft = fine_tuning.with_streaming_response
  with stream_ft.jobs.list() as response:
      for chunk in response.iter_content(chunk_size=1024):
          process(chunk)  # Handle streamed content.
  ```

**AsyncFineTuning (AsyncAPIResource)**
--------------------------------------

The asynchronous counterpart of `FineTuning`. All properties mirror the synchronous API but are designed for `async`/`await` patterns.

### **Properties**

#### **`jobs`**
```python
@cached_property
def jobs(self) -> AsyncJobs:
    return AsyncJobs(self._client)
```
* **Type:** `AsyncJobs`
* **Description:** Asynchronous version of `Jobs`. Manage fine-tuning jobs asynchronously.
* **Example:**
  ```python
  async_fine_tuning = AsyncFineTuning(client)
  jobs = async_fine_tuning.jobs
  await jobs.list()  # Async operation.
  ```

#### **`with_raw_response`** and **`with_streaming_response`**
```python
@cached_property
def with_raw_response(self) -> AsyncFineTuningWithRawResponse:
    # ...

@cached_property
def with_streaming_response(self) -> AsyncFineTuningWithStreamingResponse:
    # ...
```
* **Behavior:** Identical to their synchronous counterparts but operate asynchronously.
* **Types:** `AsyncFineTuningWithRawResponse`, `AsyncFineTuningWithStreamingResponse`
* **Examples:**
  ```python
  raw_async_ft = async_fine_tuning.with_raw_response
  response = await raw_async_ft.jobs.list()  # Async raw response.

  stream_async_ft = async_fine_tuning.with_streaming_response
  async with stream_async_ft.jobs.list() as response:
      async for chunk in response.content:
          await process_chunk(chunk)  # Async streaming.
  ```

**FineTuningWithRawResponse**
-----------------------------

Wrapper for accessing **raw HTTP responses** in synchronous fine-tuning operations.

### **Properties**

#### **`jobs`**
```python
@cached_property
def jobs(self) -> JobsWithRawResponse:
    return JobsWithRawResponse(self._fine_tuning.jobs)
```
* **Type:** `JobsWithRawResponse`
* **Description:** Apply `with_raw_response` behavior to `jobs` operations.
* **Example:**
  ```python
  raw_ft = fine_tuning.with_raw_response
  raw_jobs = raw_ft.jobs
  response = raw_jobs.retrieve(job_id="ft-123")
  print(response.status_code)  # HTTP status code.
  ```

**AsyncFineTuningWithRawResponse**
----------------------------------

Asynchronous version of `FineTuningWithRawResponse`.

### **Properties**

#### **`jobs`**
```python
@cached_property
def jobs(self) -> AsyncJobsWithRawResponse:
    return AsyncJobsWithRawResponse(self._fine_tuning.jobs)
```
* **Type:** `AsyncJobsWithRawResponse`
* **Description:** Async wrapper for raw responses in `jobs`.
* **Example:**
  ```python
  raw_async_ft = async_fine_tuning.with_raw_response
  raw_async_jobs = raw_async_ft.jobs
  response = await raw_async_jobs.retrieve(job_id="ft-123")
  print(response.headers["x-rate-limit-remaining"])  # Inspect headers.
  ```

**FineTuningWithStreamingResponse**
-----------------------------------

Enables **streamed responses** for synchronous operations.

### **Properties**

#### **`jobs`**
```python
@cached_property
def jobs(self) -> JobsWithStreamingResponse:
    return JobsWithStreamingResponse(self._fine_tuning.jobs)
```
* **Type:** `JobsWithStreamingResponse`
* **Description:** Stream responses for `jobs` operations.
* **Example:**
  ```python
  stream_ft = fine_tuning.with_streaming_response
  with stream_ft.jobs.list() as response:
      for line in response.iter_lines():
          print(line.decode())  # Process streamed JSON lines.
  ```

**AsyncFineTuningWithStreamingResponse**
----------------------------------------

Asynchronous streamed responses.

### **Properties**

#### **`jobs`**
```python
@cached_property
def jobs(self) -> AsyncJobsWithStreamingResponse:
    return AsyncJobsWithStreamingResponse(self._fine_tuning.jobs)
```
* **Type:** `AsyncJobsWithStreamingResponse`
* **Description:** Async streamed `jobs` responses.
* **Example:**
  ```python
  stream_async_ft = async_fine_tuning.with_streaming_response
  async with stream_async_ft.jobs.list() as response:
      async for line in response.content:
          print(line.decode())  # Async streamed content.
  ```

**Comparison Table**
--------------------

| **Class**                          | **Sync/Async** | **Purpose**                                      |
|-------------------------------------|----------------|--------------------------------------------------|
| `FineTuning`                       | Sync           | Core fine-tuning operations.                     |
| `AsyncFineTuning`                  | Async          | Async version of `FineTuning`.                   |
| `FineTuningWithRawResponse`        | Sync           | Raw HTTP responses (e.g., headers, status).      |
| `AsyncFineTuningWithRawResponse`   | Async          | Async raw responses.                             |
| `FineTuningWithStreamingResponse`  | Sync           | Streamed responses (large data, real-time).      |
| `AsyncFineTuningWithStreamingResponse` | Async      | Async streamed responses.                        |

**Best Practices**
------------------

1. **Use `with_raw_response`** when debugging or needing HTTP metadata.
2. **Prefer `with_streaming_response`** for large datasets or real-time processing.
3. **Async API** for non-blocking I/O-bound operations (e.g., web servers, background jobs).
4. Always handle streamed responses with `with` blocks to ensure proper resource cleanup.

**Error Handling**
------------------

- **Synchronous:** Wrap calls in `try`-`except` blocks for `HTTPError`, `Timeout`, or `ConnectionError`.
- **Asynchronous:** Use `try`-`await`-`except` for `AsyncHTTPError`, etc.

Example:
```python
try:
    jobs = fine_tuning.jobs.list()
except HTTPError as e:
    print(e.response.status_code)

# Async
try:
    jobs = await async_fine_tuning.jobs.list()
except AsyncHTTPError as e:
    print(await e.response.text())
```

By following this documentation, developers can proficiently leverage the Fine-Tuning API for both synchronous and asynchronous workflows, ensuring robust, efficient, and scalable model customization.