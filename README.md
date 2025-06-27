# Multi-Agent System with Tool Calling

Built a multi-agent system using **LangChain**, **LangGraph**, and **Pydantic** with **ChatGPT/Anthropic** as the underlying language model. Inclusive of automated research, fact-checking, and content creation workflows via modular AI agents.

## Features

- **Multi-Agent Coordination by Tool Calling**: Research and Writing agents working together
   1. **Research Agent**: Discovers and summarizes high-quality sources on your topic.
   2. **Fact Checking Agent**: Verifies claims against trusted references to ensure accuracy.
   3. **Content Generation Agent**: Produces human-readable summaries, reports, presentations, or blog posts.
   4. **Quality Assurance Agent**: Reviews generated content for coherence, style consistency, and audience suitability.
- **Structured Data**: Pydantic models for type-safe data handling
- **Workflow Orchestration**: LangGraph for managing agent interactions
- **State Management**: Shared state between agents with checkpointing
- **RESTful API Interface**: Start, monitor, cancel pipelines; fetch results via JSON endpoints.
- **Real-Time Progress Tracking**: Background execution with callbacks reporting agent-level progress.
- **Streamlit Dashboard**: User-friendly UI to orchestrate pipelines without writing code.
- **Comprehensive Logging**: Timestamped logs to both console and logs/research_pipeline.log.

## Architecture

```
┌───────────────────┐       ┌─────────────────────────┐
│  Client (cURL,    │  HTTP │   FastAPI Application   │
│  Streamlit UI)    ├──────▶│  ┌───────────────────┐  │
└───────────────────┘       │  │ ResearchPipeline  │  │
                            │  │  ┌─────────────┐  │  │
                            │  │  │ Agents      │  │  │
                            │  │  │  • Research │  │  │
                            │  │  │  • FactCheck│  │  │
                            │  │  │  • Generate │  │  │
                            │  │  │  • QA       │  │  │
                            │  │  └─────────────┘  │  │
                            │  └───────────────────┘  │
                            │    BackgroundTasks      │
                            └─────────────────────────┘

```
1. FastAPI exposes endpoints under ```/research/...```
2. BackgroundTasks launch ```execute_pipeline()``` asynchronously.
3. ResearchPipeline (in ```multi_agent_system.py```) coordinates each agent, reporting back via a progress_callback.
4. In-Memory Store (```pipeline_results```, ```active_pipelines```) holds pipeline state until retrieval.
5. Streamlit App (```st_app.py```) calls these endpoints to provide a dashboard.

## Project Structure

```
MAS-ResearchTool/
├── logs/                     # Application logs
│   └── research_pipeline.log
├── outputs/                  # Pipeline output artifacts
├── pyproject.toml            # Packaging & dependencies (PDM)
├── requirements.txt          # pip-installable dependencies
├── pdm.lock                  # PDM lock file
├── uv.lock                   # uv project lock
├── src/
│   ├── __init__.py
│   ├── main.py               # FastAPI entry point
│   ├── multi_agent_system.py # Core pipeline & agents
│   └── st_app.py                # Streamlit dashboard
├── tests/                    # Unit & integration tests
└── README.md                 # This file
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/sree-sphere/MAS-ResearchTool.git
   cd MAS-ResearchTool
   ```

2. **Setup virtual env**
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```

3. **Install Dependencies**
   Using uv
   ```bash
   uv pip install -r requirements.txt
   uv pip install -e .
   ```
   Using pip
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API Key** (Or Anthropic. API_URL is optional)
   ```bash
   cp .env.example .env
   export OPENAI_API_KEY=<openai_key>
   export ANTHROPIC_API_KEY= <anthropic_key>
   export API_URL=http://localhost:8000
   ```

## Usage

1. **FastAPI server**
   ```bash
   uvicorn src.main:app --reload
   ```
   - Docs: http://localhost:8000/docs

2. **Streamlit**
   ```bash
   streamlit run src/app.py
   ```
   - Dashboard: http://localhost:8501

## Key Technologies

- **LangChain**: for building the application with LLM
- **LangGraph**: for building stateful, multi-actor application
- **Pydantic**: Data validation using Python type annotations
- **OpenAI GPT**: for intelligent responses
- **asyncio & BackgroundTasks**: for non-blocking execution

## Workflow Process

1. User submits a research request (```topic```, ```depth```, ```content_types```, etc.).
2. FastAPI generates a ```pipeline_id``` and returns an estimated completion timestamp.
3. Background task ```execute_pipeline()``` starts:
   - Updates status to ```running``` (5%).
   - Invokes each agent in sequence with a progress callback.
   - Aggregates partial outputs and metrics.
4. On completion:
   - Saves final content and agent metrics.
   - Marks status ```completed```, progress 100%.
5. Client fetches ```/research/results/{pipeline_id}``` to retrieve JSON payload.

## Sample Output

When you run the app, you'll see:

```json
// GET /research/results/pipeline_20250627_142012_1234
{
  "pipeline_id": "pipeline_20250627_142012_1234",
  "status": "completed",
  "results": {
    "summary": "Artificial Intelligence in Healthcare is ...",
    "report": "...",
    "presentation": "...",
    "blog_post": "..."
  },
  "completed_at": "2025-06-27T14:35:10.123456",
  "execution_time": 1320.45,
  "agent_metrics": {
    "ResearchAgent": { "duration": 300.12, "sources_used": 10 },
    "FactCheckAgent": { "duration": 150.34, "errors_found": 2 },
    "ContentGenAgent": { "duration": 500.78 },
    "QAAgent": { "duration": 369.21, "issues_flagged": 1 }
  }
}
```

## Error Handling

The system includes:
- JSON parsing with fallback data structures
- Tool error handling and recovery
- State validation with Pydantic models

## Customization

### Adding New Agents
1. **Create** a new agent class in ```src/multi_agent_system.py```, subclassing the common base (BaseAgent).
2. **Implement** a ```run()``` method accepting ```ResearchRequest``` and ```progress_callback```.
3. **Register** your agent:
   ```python
   self.agents.append(MyNewAgent(...))
   ```
4. **Adjust** configuration schema in ```PipelineRequest``` if you need new parameters.
5. **Rebuild** and restart the server.

## Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Pydantic Documentation](https://pydantic.dev/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## Contributing

Feel free to:
- Extend the agents with new capabilities
- Add more sophisticated tools
- Implement real API integrations
- Enhance the workflow with additional steps

You can start by:
1. Fork the repository.
2. Create a feature branch: ```git checkout -b feature/my-agent```.
3. Submit a pull request with clear description of changes.

Unit tests are yet pending.

## License

Distributed under the MIT License. See LICENSE for details.