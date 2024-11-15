# Project SMILE Backend

## Concept
Project SMILE is an AI-powered chat application that features a witty and engaging AI assistant named Smiles. The project utilizes advanced language models and tools to provide an interactive and helpful conversational experience.

## Features
- Interactive CLI-based chat interface
- Streaming responses for real-time interaction
- Web search capabilities using Tavily Search
- File management tools
- Support for multiple LLM providers (OpenAI and Anthropic)
- Configurable LLM settings and parameters
- Markdown rendering for better text formatting

## Project Details

### API
The backend API is built using FastAPI with the following features:
- CORS middleware enabled for cross-origin requests
- RESTful endpoints (to be implemented)
- API router structure for organized endpoint management

### Services
1. **Core Chat Service** (Reference: ```python:app/smile.py
startLine: 15
endLine: 85



## Roadmap
- [ ] Implement RESTful endpoints
- [ ] Implement Web search capabilities using Tavily Search
- [ ] Implement File management tools
- [ ] implement Code Interpreter
- [ ] Implement support for multiple LLM providers (OpenAI and Anthropic)
- [ ] Implement Configurable LLM settings and parameters
- [ ] Implement Markdown rendering for better text formatting
- [ ] Implement Streaming responses for real-time interaction
- [ ] Implement Neo4j database for knowledge base
- [ ] implement Mode selection: 
    - [ ] Focus Mode
    - [ ] Creative Mode
    - [ ] Funtime Mode
    - [ ] Therapist Mode
    - [ ] Custom Mode
- [ ] Implement Web UI
- [ ] Implement Communication channels:
    - [ ] CLI
    - [ ] Web UI
    - [ ] API
    - [ ] Webhook (it would interesting to have Smiles react autonomously to external events.)
    - [ ] Sms (should have its own phone number for SMS exchanges. Will use Twilio)
    - [ ] Email (should have its own email address for email exchanges. But have access to the user's email account for management)
    - [ ] Voice TTS / STT
    - [ ] Voice Real-time
    - [ ] Voice Phone Call (will use Twilio)
- [ ] Implement Webhook for web integration
- [ ] Implement API Key for security
- [ ] Implement Logging for monitoring and debugging
- [ ] Implement Containerization:
    - [ ] Docker
    - [ ] Docker Compose
- [ ] Implement CI/CD Pipeline
