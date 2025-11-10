from google.adk.agents import Agent
from google.adk.tools import google_search  # Import the tool

root_agent = Agent(
   name="basic_search_agent",
   model="gemini-2.5-flash",
   description="Agent to answer questions using Google Search.",
   instruction="You are an expert researcher. You always stick to the facts.",
   tools=[google_search]
)

# Execute once before running the agent for the first time:
# export SSL_CERT_FILE="$(python -m certifi)"
