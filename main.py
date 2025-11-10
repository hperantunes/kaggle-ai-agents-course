import asyncio
import os
from typing import Final

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search


PROMPT: Final[str] = (
    "What is Agent Development Kit from Google? What languages is the SDK available in?"
)


def load_api_key(env_var: str = "GOOGLE_API_KEY", dotenv_path: str = ".env") -> str:
    """Load the API key from the environment, raising if it is missing."""
    load_dotenv(dotenv_path)
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"Environment variable {env_var} is not set.")
    print("API key successfully loaded.")
    return api_key


def create_agent() -> Agent:
    """Instantiate the root agent used for answering questions."""
    agent = Agent(
        name="helpful_assistant",
        model="gemini-2.5-flash-lite",
        description="A simple agent that can answer general questions.",
        instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
        tools=[google_search],
    )
    print("Root agent successfully configured.")
    return agent


def create_runner(agent: Agent) -> InMemoryRunner:
    """Return an in-memory runner configured for the supplied agent."""
    runner = InMemoryRunner(agent=agent, app_name="agents")
    print("Runner successfully configured.")
    return runner


async def run_prompt(prompt: str, runner: InMemoryRunner) -> None:
    """Execute the prompt against the runner and display the response."""
    response = await runner.run_debug(prompt)
    print(response)


async def main() -> None:
    _ = load_api_key()
    agent = create_agent()
    runner = create_runner(agent)
    await run_prompt(PROMPT, runner)


if __name__ == "__main__":
    asyncio.run(main())
