# LiveKit Plugin SMILES

A LiveKit plugin that integrates the SMILES API into LiveKit's agent framework, enabling voice interactions with the SMILES backend service.

## Installation

### Local Installation
```bash
pip install -e .
```

### Docker Installation
1. Configure your environment variables:
```bash
# Edit .env with your credentials
# Make sure LIVEKIT_URL points to your cloud LiveKit server
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

## Configuration

The plugin requires the following environment variables in your `.env` file:

```env
SMILES_API_URL=http://localhost:8000  # URL of your SMILES API
SMILES_API_KEY=your_api_key           # Optional: API key for authentication
LIVEKIT_URL=wss://your-livekit-cloud  # Your cloud LiveKit server URL
LIVEKIT_API_KEY=your_livekit_key      # LiveKit API key
LIVEKIT_API_SECRET=your_livekit_secret # LiveKit API secret
OPENAI_API_KEY=your_openai_key        # Required for TTS
DEEPGRAM_API_KEY=your_deepgram_key    # Required for STT
```

## Usage

### Running with Docker (Recommended)
The Docker setup is configured to connect to your cloud LiveKit server:
```bash
docker-compose up
```

### Running Locally
1. Set up the required environment variables
2. Run the example SMILES assistant:
```bash
python examples/smiles_assistant.py
```

## Components

### SMILES LLM

The main component is the SMILES LLM implementation that interfaces with your SMILES API:

```python
from livekit_plugin_smiles import LLM as SmilesLLM

llm = SmilesLLM(
    api_url="http://localhost:8000",
    api_key="your_api_key"  # optional
)
```

### Voice Pipeline Integration

The SMILES LLM can be used in LiveKit's VoicePipelineAgent:

```python
agent = VoicePipelineAgent(
    vad=silero.VAD.load(),
    stt=deepgram.STT(model="nova-2-general"),
    llm=SmilesLLM(api_url="your_api_url"),
    tts=openai.TTS(),
    turn_detector=turn_detector.EOUModel(),
    chat_ctx=initial_ctx,
)
```

## Development

### Local Development
1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Configure your `.env` file with cloud LiveKit server details
4. Run the example assistant for testing

### Docker Development
1. Make changes to the code
2. Rebuild and run the container:
   ```bash
   docker-compose up --build
   ```

## Architecture

The plugin integrates with LiveKit's agent framework by:
1. Implementing the LLM interface required by LiveKit
2. Converting between LiveKit's chat context format and SMILES API format
3. Handling streaming responses from the SMILES API
4. Managing the connection to the SMILES backend service

The voice pipeline uses:
- Silero VAD for voice activity detection
- Deepgram for speech-to-text
- SMILES for language model processing
- OpenAI for text-to-speech
- LiveKit's turn detection for conversation management

## Error Handling

The plugin includes robust error handling for:
- API connection issues
- Response processing errors
- Session management
- Resource cleanup

Errors are logged and graceful fallback responses are provided to ensure a smooth user experience.

## Docker Environment

The Docker setup is designed to be simple and efficient:
- Python environment with essential dependencies
- Automatic environment variable loading from `.env`
- Direct connection to cloud LiveKit server
- Optimized for production use
