import json
import logging
import os

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search

from google.genai import types
from typing import Any, Iterable, List

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"🧹 Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ---- Handle potential incorrect datatype returned by the agent ----
def _normalize_papers(raw_papers: Any) -> List[str]:
    """Normalize tool input into a clean list of paper titles."""

    def _split_lines(text: str) -> List[str]:
        cleaned = []
        for line in text.splitlines():
            entry = line.strip(" -\t•")
            if entry:
                cleaned.append(entry)
        return cleaned

    normalized: List[str] = []

    if isinstance(raw_papers, str):
        candidate = raw_papers.strip()
        if not candidate:
            return []
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            normalized.extend(_split_lines(candidate))
        else:
            if isinstance(parsed, list):
                for item in parsed:
                    entry = str(item).strip()
                    if entry:
                        normalized.append(entry)
            else:
                normalized.extend(_split_lines(candidate))
        return normalized

    if not isinstance(raw_papers, Iterable):
        return normalized

    for item in raw_papers:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "\n" in text:
                normalized.extend(_split_lines(text))
            else:
                normalized.append(text)
        else:
            if isinstance(parsed, list):
                for entry in parsed:
                    clean_entry = str(entry).strip()
                    if clean_entry:
                        normalized.append(clean_entry)
            else:
                normalized.extend(_split_lines(text))

    return normalized


def count_papers(papers: List[str]):
    """Count how many distinct paper titles were discovered.

    The agent occasionally sends a single string (e.g. bullet list or JSON) or
    a sequence of strings. We coerce the value into a list before counting to
    ensure the total reflects the actual number of papers.
    """

    normalized_papers = _normalize_papers(papers)
    logging.debug("Normalized papers: %s", normalized_papers)
    return len(normalized_papers)


# Google Search agent
google_search_agent = LlmAgent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for information using Google search",
    instruction="""Use the google_search tool to find information on the given topic. Return the raw search results.
    If the user asks for a list of papers, then give them the list of research papers you found and not the summary.""",
    tools=[google_search]
)


# Root agent
root_agent = LlmAgent(
    name="research_paper_finder_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Your task is to find research papers and count them. 

    You MUST ALWAYS follow these steps:
    1) Find research papers on the user provided topic using ONLY the 'google_search_agent'. 
    2) Whenever new papers are found, cleanly structure the results as a JSON array of paper titles and immediately call 'count_papers'.
    3) Never skip step 2. Your final response MUST include both the JSON list and the total number of papers returned by 'count_papers'.
    """,
    tools=[AgentTool(agent=google_search_agent), count_papers]
)

# Run agent in web mode with:
# adk web --log_level DEBUG
# Use in the prompt: "Find latest quantum computing papers"
