# Project SMILE
## Smart Management of Integrated Life Entropy

Project SMILE is a sophisticated personal AI assistant designed to transform life's daily chaos into seamless order. Built with advanced cognitive architecture, SMILE combines state-of-the-art memory systems, natural language processing, and adaptive communication to create a truly personalized assistant experience.

## Core Features

### 1. Cognitive Memory System
- **Long-term Memory Storage**: Sophisticated memory architecture using Neo4j graph database
- **Multi-dimensional Memory Types**: Support for various memory categories including:
  - Personal information and preferences
  - Events and temporal data
  - Organizations and locations
  - Tasks and projects
  - Historical context
- **Memory Validation**: Built-in validation system to maintain memory accuracy and resolve contradictions
- **Temporal Context Tracking**: Advanced handling of time-based information including:
  - Recurring events
  - Valid time periods
  - Historical references

### 2. Natural Language Understanding
- **Context-Aware Processing**: Maintains conversation context across interactions
- **Entity Recognition**: Identifies and tracks mentions of people, organizations, and locations
- **Semantic Analysis**: Extracts keywords, categories, and cognitive aspects from conversations
- **Sentiment Analysis**: Tracks emotional context and importance of interactions

### 3. Knowledge Graph Integration
- **Relationship Tracking**: Maps connections between entities, memories, and concepts
- **Semantic Network**: Builds a rich network of interconnected information
- **Association Strength**: Tracks and updates relationship strengths over time
- **Graph-based Querying**: Enables complex relationship-based information retrieval

### 4. Document Management
- **Smart Document Organization**: Automatic categorization and tagging of documents
- **Content Analysis**: Extracts key information and topics from documents
- **Version Control**: Tracks document versions and access patterns
- **Metadata Management**: Rich metadata tracking including:
  - Creation and access timestamps
  - Language detection
  - Content summaries
  - Custom tags and categories

### 5. Modern Web Interface
- **Responsive Design**: Built with Next.js and Tailwind CSS
- **Dark/Light Mode**: Automatic theme switching based on system preferences
- **Custom Typography**: Optimized reading experience with Geist font family
- **Interactive UI Components**: Modern, accessible interface components

## Technical Architecture

### Backend Stack
- **Python 3.8+**: Core application runtime
- **FastAPI**: High-performance API framework
- **Neo4j**: Graph database for memory and relationship storage
- **PostgreSQL**: Relational database for structured data
- **LangChain**: Framework for LLM integration
- **OpenAI/Anthropic**: AI model providers

### Frontend Stack
- **Next.js 15+**: React framework for web interface
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Shadcn/UI**: Modern UI component library
- **React Markdown**: Rich text rendering

### Infrastructure
- **Docker**: Containerized deployment
- **Docker Compose**: Multi-container orchestration
- **Poetry**: Python dependency management
- **Environment Management**: Flexible configuration via .env files

## Setup Instructions

**Note**: For detailed backend setup instructions and additional information, please refer to the [backend README](backend/README.md).

Follow these steps to set up Project SMILE on your local machine for development and testing purposes.

### Prerequisites

- **Docker**: Ensure Docker is installed on your machine. [Download Docker here](https://www.docker.com/get-started).
- **Python 3.8+**: Project SMILE requires Python version 3.8 or higher.
- **Poetry**: We use Poetry for dependency management. Install it by following the [official guide](https://python-poetry.org/docs/#installation).

### Clone the Repository

Clone the Project SMILE repository to your local machine:

```bash
git clone https://github.com/elampron/Project-SMILE.git
cd project-smile
```

### Start the PostgreSQL Database

Project SMILE relies on a PostgreSQL database. Use Docker Compose to set up the database container.


- **Start** the PostgreSQL container:

  ```bash
  docker-compose up -d
  ```

  This command runs the PostgreSQL service in the background.
  The database will be available at `localhost:5432`.
  The data files are stored in `db/data`. Inlcuded in gitignore.

### Configure Environment Variables

Set up your environment variables by copying the example file and editing it:

```bash
cp .env.example .env
```

- **Edit** `.env` and provide the necessary API keys and settings. At minimum, you should provide:

  ```dotenv
  OPENAI_API_KEY=your-openai-api-key
  ANTHROPIC_API_KEY=your-anthropic-api-key (optional)
  ```

### Install Dependencies

Install the required Python packages using Poetry:

- **Navigate** to the backend directory:

  ```bash
  cd backend
  ```

- **Install** dependencies:

  ```bash
  poetry install
  ```

  This will create a virtual environment and install all necessary packages.

### Activate the Virtual Environment

Activate the Poetry-managed virtual environment:

```bash
poetry shell
```

### Run the Assistant via CLI

You can test SMILE using the command-line interface provided in `app.utils.cli`:

- **Run** the CLI utility:

  ```bash
  python -m app.utils.cli
  ```

- **Interact** with SMILE:

  - The CLI will prompt you to enter commands or messages.
  - Type your input and press Enter to communicate with the assistant.

### Stop the PostgreSQL Container (Optional)

When you're done testing, you can stop the PostgreSQL container:

```bash
docker-compose down
```

---

## Notes

- **Docker Dependency**: Docker is required to run the PostgreSQL database container, ensuring consistency across development environments.
- **Database Setup**: The default configuration connects to a PostgreSQL database running on `localhost` with the default credentials provided in `app/configs/app_config.yaml`. Adjust these settings if necessary.
- **Additional Information**: For more advanced setup options, troubleshooting, and detailed backend configurations, please see the [backend README](backend/README.md).
- **Future Development**: As the project evolves, additional setup steps such as database migrations or frontend builds may be required. These will be documented accordingly.
- **Feedback and Contributions**: Feel free to open issues or submit pull requests. Your contributions are welcome!

---

## Troubleshooting

- **Docker Issues**: If you encounter problems starting the Docker container, ensure Docker Engine is running and you have the necessary permissions.
- **Dependency Problems**: Should you face issues with dependencies, try updating Poetry and reinstalling:

  ```bash
  poetry self update
  poetry install --no-cache
  ```

- **API Key Errors**: Make sure your API keys are correct and have the necessary permissions.

---

## Contact

For questions or support, please contact [your-email@example.com](mailto:your-email@example.com).

## Advanced Features

### Memory Categories
- **Person Memory**: Stores individual profiles, preferences, and interaction history
- **Organization Memory**: Tracks business and institutional information
- **Location Memory**: Geographic and spatial context storage
- **Event Memory**: Temporal data with support for recurring events
- **Task/Project Memory**: Progress tracking and deadline management
- **Conversation Context**: Maintains interaction continuity
- **Product Information**: Tracks product details and relationships
- **Interest/Hobbies**: Personal preference and interest tracking

### Cognitive Processing
- **Multi-aspect Analysis**: Processes factual, emotional, and temporal aspects
- **Confidence Scoring**: Maintains confidence levels for stored information
- **Contradiction Resolution**: Identifies and resolves conflicting information
- **Association Building**: Creates and maintains memory relationships
- **Temporal Reasoning**: Handles complex time-based relationships

## Development Guidelines

### Code Standards
- Comprehensive documentation and docstrings
- Detailed logging throughout the application
- Type hints and validation
- Error handling and graceful degradation

### Security
- API key management
- Secure data storage
- Access control and validation
- Privacy-first design