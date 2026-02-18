**Audio API Documentation**
==========================

**Overview**
------------

The Audio API provides a set of classes for interacting with audio-related resources. The main classes are `Audio` and `AsyncAudio`, which serve as the entry points for synchronous and asynchronous operations, respectively.

**Main API Classes**
--------------------

### **`Audio` Class** ( **SYNC** )

**Inheritance:** `SyncAPIResource`

**Description:** The `Audio` class provides synchronous access to audio-related resources, including transcriptions, translations, and speech.

**Attributes:**

*   **`transcriptions`** (`Transcriptions`): A cached property that returns a `Transcriptions` object for interacting with transcription resources.
*   **`translations`** (`Translations`): A cached property that returns a `Translations` object for interacting with translation resources.
*   **`speech`** (`Speech`): A cached property that returns a `Speech` object for interacting with speech resources.
*   **`with_raw_response`** (`AudioWithRawResponse`): A cached property that returns an `AudioWithRawResponse` object, allowing you to access raw response data (e.g., headers) for any HTTP method call.
*   **`with_streaming_response`** (`AudioWithStreamingResponse`): A cached property that returns an `AudioWithStreamingResponse` object, providing an alternative to `.with_raw_response` that doesn't eagerly read the response body.

**Example Usage:**
```python
from openai import Audio

audio = Audio(client)  # assuming 'client' is your API client instance

# Access transcriptions resource
transcriptions = audio.transcriptions
# Use transcriptions object to perform operations (e.g., create transcription)
transcription_result = transcriptions.create(...)

# Access raw response data for any HTTP method call
raw_audio = audio.with_raw_response
raw_transcription_result = raw_audio.transcriptions.create(...)
print(raw_transcription_result.headers)
```

### **`AsyncAudio` Class** ( **ASYNC** )

**Inheritance:** `AsyncAPIResource`

**Description:** The `AsyncAudio` class provides asynchronous access to audio-related resources, including transcriptions, translations, and speech.

**Attributes:**

*   **`transcriptions`** (`AsyncTranscriptions`): A cached property that returns an `AsyncTranscriptions` object for interacting with transcription resources asynchronously.
*   **`translations`** (`AsyncTranslations`): A cached property that returns an `AsyncTranslations` object for interacting with translation resources asynchronously.
*   **`speech`** (`AsyncSpeech`): A cached property that returns an `AsyncSpeech` object for interacting with speech resources asynchronously.
*   **`with_raw_response`** (`AsyncAudioWithRawResponse`): A cached property that returns an `AsyncAudioWithRawResponse` object, allowing you to access raw response data (e.g., headers) for any HTTP method call asynchronously.
*   **`with_streaming_response`** (`AsyncAudioWithStreamingResponse`): A cached property that returns an `AsyncAudioWithStreamingResponse` object, providing an alternative to `.with_raw_response` that doesn't eagerly read the response body asynchronously.

**Example Usage:**
```python
import asyncio
from openai import AsyncAudio

async def main():
    async_audio = AsyncAudio(client)  # assuming 'client' is your API client instance

    # Access transcriptions resource asynchronously
    transcriptions = async_audio.transcriptions
    # Use transcriptions object to perform operations (e.g., create transcription) asynchronously
    transcription_result = await transcriptions.create(...)

    # Access raw response data for any HTTP method call asynchronously
    raw_async_audio = async_audio.with_raw_response
    raw_transcription_result = await raw_async_audio.transcriptions.create(...)
    print(raw_transcription_result.headers)

asyncio.run(main())
```

**Raw Response Classes**
------------------------

The following classes provide access to raw response data (e.g., headers) for HTTP method calls:

### **`AudioWithRawResponse` Class**

**Description:** This class is used as a prefix for any HTTP method call on the `Audio` class to return the raw response object instead of the parsed content.

**Attributes:**

*   **`transcriptions`** (`TranscriptionsWithRawResponse`): A cached property that returns a `TranscriptionsWithRawResponse` object.
*   **`translations`** (`TranslationsWithRawResponse`): A cached property that returns a `TranslationsWithRawResponse` object.
*   **`speech`** (`SpeechWithRawResponse`): A cached property that returns a `SpeechWithRawResponse` object.

**Initialization:**
```python
raw_audio = AudioWithRawResponse(audio_instance)
```

**Example Usage:**
```python
raw_audio = audio.with_raw_response  # from Audio class instance
raw_transcription_result = raw_audio.transcriptions.create(...)
print(raw_transcription_result.headers)
```

### **`AsyncAudioWithRawResponse` Class**

**Description:** This class is the asynchronous counterpart of `AudioWithRawResponse`, used with the `AsyncAudio` class.

**Attributes:**

*   **`transcriptions`** (`AsyncTranscriptionsWithRawResponse`): A cached property that returns an `AsyncTranscriptionsWithRawResponse` object.
*   **`translations`** (`AsyncTranslationsWithRawResponse`): A cached property that returns an `AsyncTranslationsWithRawResponse` object.
*   **`speech`** (`AsyncSpeechWithRawResponse`): A cached property that returns an `AsyncSpeechWithRawResponse` object.

**Initialization:**
```python
raw_async_audio = AsyncAudioWithRawResponse(async_audio_instance)
```

**Example Usage:**
```python
raw_async_audio = async_audio.with_raw_response  # from AsyncAudio class instance
raw_transcription_result = await raw_async_audio.transcriptions.create(...)
print(raw_transcription_result.headers)
```

**Streaming Response Classes**
-----------------------------

The following classes provide an alternative to the raw response classes, allowing you to work with streaming responses:

### **`AudioWithStreamingResponse` Class**

**Description:** This class is similar to `AudioWithRawResponse` but doesn't eagerly read the response body, providing a streaming approach.

**Attributes:**

*   **`transcriptions`** (`TranscriptionsWithStreamingResponse`): A cached property that returns a `TranscriptionsWithStreamingResponse` object.
*   **`translations`** (`TranslationsWithStreamingResponse`): A cached property that returns a `TranslationsWithStreamingResponse` object.
*   **`speech`** (`SpeechWithStreamingResponse`): A cached property that returns a `SpeechWithStreamingResponse` object.

**Initialization:**
```python
streaming_audio = AudioWithStreamingResponse(audio_instance)
```

**Example Usage:**
```python
streaming_audio = audio.with_streaming_response  # from Audio class instance
streaming_transcription_result = streaming_audio.transcriptions.create(...)
for chunk in streaming_transcription_result:
    print(chunk)
```

### **`AsyncAudioWithStreamingResponse` Class**

**Description:** The asynchronous version of `AudioWithStreamingResponse`, used with the `AsyncAudio` class.

**Attributes:**

*   **`transcriptions`** (`AsyncTranscriptionsWithStreamingResponse`): A cached property that returns an `AsyncTranscriptionsWithStreamingResponse` object.
*   **`translations`** (`AsyncTranslationsWithStreamingResponse`): A cached property that returns an `AsyncTranslationsWithStreamingResponse` object.
*   **`speech`** (`AsyncSpeechWithStreamingResponse`): A cached property that returns an `AsyncSpeechWithStreamingResponse` object.

**Initialization:**
```python
streaming_async_audio = AsyncAudioWithStreamingResponse(async_audio_instance)
```

**Example Usage:**
```python
streaming_async_audio = async_audio.with_streaming_response  # from AsyncAudio class instance
streaming_transcription_result = await streaming_async_audio.transcriptions.create(...)
async for chunk in streaming_transcription_result:
    print(chunk)
```

**Additional Resources**
------------------------

For more information on accessing raw response data or using streaming responses, refer to the following documentation:

*   [Accessing Raw Response Data (e.g., Headers)](https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers)
*   [Using Streaming Responses](https://www.github.com/openai/openai-python#with_streaming_response)

**API Reference Index**
----------------------

| Class | Description | Async/Sync |
| --- | --- | --- |
| `Audio` | Main synchronous entry point for audio resources. | SYNC |
| `AsyncAudio` | Main asynchronous entry point for audio resources. | ASYNC |
| `AudioWithRawResponse` | Access raw response data for `Audio` class HTTP calls. | SYNC |
| `AsyncAudioWithRawResponse` | Access raw response data for `AsyncAudio` class HTTP calls. | ASYNC |
| `AudioWithStreamingResponse` | Use streaming responses with `Audio` class HTTP calls. | SYNC |
| `AsyncAudioWithStreamingResponse` | Use streaming responses with `AsyncAudio` class HTTP calls. | ASYNC |

Each of the `Transcriptions`, `Translations`, `Speech`, and their async counterparts (`AsyncTranscriptions`, `AsyncTranslations`, `AsyncSpeech`) have their own detailed documentation, which can be accessed through the respective attributes on the `Audio` or `AsyncAudio` instances.

Ensure you explore those resources for comprehensive details on available methods and parameters for interacting with audio-related functionalities.