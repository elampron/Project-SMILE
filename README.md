# Project SMILE
## Smart Management of Integrated Life Entropy

Project SMILE is a personal AI assistant designed to transform life’s daily chaos into seamless order. Built to be witty, sociable, and a little bit nerdy, SMILE combines advanced memory, decision-making, and conversation tools to help manage emails, schedules, and everyday tasks. With a focus on adaptive communication, SMILE can switch between modes like “Focus” for productivity or “Therapist” for support, creating a personalized experience for any need. This open-source project is built to integrate effortlessly, using science and smart architecture to bring a friendly, organized touch to life’s entropy.

---

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