# Audio API Documentation

## Core API Classes

### `Audio`

The primary class for synchronous audio processing operations. Inherits from `SyncAPIResource`.

```python
from openai import OpenAI

client = OpenAI()
audio = client.audio  # Returns an instance of Audio
```

#### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `transcriptions` | `Transcriptions` | Provides access to audio transcription functionality |
| `translations` | `Translations` | Provides access to audio translation functionality |
| `speech` | `Speech` | Provides access to speech synthesis functionality |
| `with_raw_response` | `AudioWithRawResponse` | Returns raw HTTP responses instead of parsed content |
| `with_streaming_response` | `AudioWithStreamingResponse` | Alternative that doesn't eagerly read response bodies |

### `AsyncAudio`

The primary class for asynchronous audio processing operations. Inherits from `AsyncAPIResource`.

```python
from openai import AsyncOpenAI
import asyncio

async def process_audio():
    client = AsyncOpenAI()
    audio = client.audio  # Returns an instance of AsyncAudio
    
asyncio.run(process_audio())
```

#### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `transcriptions` | `AsyncTranscriptions` | Provides asynchronous audio transcription functionality |
| `translations` | `AsyncTranslations` | Provides asynchronous audio translation functionality |
| `speech` | `AsyncSpeech` | Provides asynchronous speech synthesis functionality |
| `with_raw_response` | `AsyncAudioWithRawResponse` | Returns raw HTTP responses instead of parsed content |
| `with_streaming_response` | `AsyncAudioWithStreamingResponse` | Alternative that doesn't eagerly read response bodies |

## Response Handling Classes

### `AudioWithRawResponse`

Wrapper for `Audio` that returns raw HTTP responses.

#### Constructor
- `__init__(audio: Audio)`: Initializes with an `Audio` instance

#### Properties
- `transcriptions`: Returns `TranscriptionsWithRawResponse`
- `translations`: Returns `TranslationsWithRawResponse`
- `speech`: Returns `SpeechWithRawResponse`

### `AsyncAudioWithRawResponse`

Wrapper for `AsyncAudio` that returns raw HTTP responses.

#### Constructor
- `__init__(audio: AsyncAudio)`: Initializes with an `AsyncAudio` instance

#### Properties
- `transcriptions`: Returns `AsyncTranscriptionsWithRawResponse`
- `translations`: Returns `AsyncTranslationsWithRawResponse`
- `speech`: Returns `AsyncSpeechWithRawResponse`

### `AudioWithStreamingResponse`

Wrapper for `Audio` that provides streaming responses.

#### Constructor
- `__init__(audio: Audio)`: Initializes with an `Audio` instance

#### Properties
- `transcriptions`: Returns `TranscriptionsWithStreamingResponse`
- `translations`: Returns `TranslationsWithStreamingResponse`
- `speech`: Returns `SpeechWithStreamingResponse`

### `AsyncAudioWithStreamingResponse`

Wrapper for `AsyncAudio` that provides streaming responses.

#### Constructor
- `__init__(audio: AsyncAudio)`: Initializes with an `AsyncAudio` instance

#### Properties
- `transcriptions`: Returns `AsyncTranscriptionsWithStreamingResponse`
- `translations`: Returns `AsyncTranslationsWithStreamingResponse`
- `speech`: Returns `AsyncSpeechWithStreamingResponse`

## Usage Examples

### Basic Transcription

```python
from openai import OpenAI

client = OpenAI()
audio = client.audio

# Transcribe an audio file
transcription = audio.transcriptions.create(
    file=open("audio.mp3", "rb"),
    model="whisper-1"
)
print(transcription.text)
```

### Using Raw Response

```python
from openai import OpenAI

client = OpenAI()

# Get raw response with headers and status code
response = client.audio.with_raw_response.transcriptions.create(
    file=open("audio.mp3", "rb"),
    model="whisper-1"
)

print(f"Status code: {response.status_code}")
print(f"Headers: {response.headers}")
print(f"Content: {response.parsed.text}")
```

### Asynchronous Speech Synthesis

```python
import asyncio
from openai import AsyncOpenAI

async def generate_speech():
    client = AsyncOpenAI()
    
    speech_response = await client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="Hello world!"
    )
    
    with open("output.mp3", "wb") as f:
        f.write(speech_response.content)

asyncio.run(generate_speech())
```

### Streaming Response

```python
from openai import OpenAI

client = OpenAI()

with open("large_audio.mp3", "rb") as audio_file:
    with client.audio.with_streaming_response.transcriptions.create(
        file=audio_file,
        model="whisper-1"
    ) as response:
        # Process the response as it streams
        for chunk in response.iter_content(chunk_size=4096):
            # Process chunk
            pass
        
        # Full response available after streaming completes
        result = response.parsed
        print(result.text)
```