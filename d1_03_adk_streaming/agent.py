from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
   name="basic_search_agent",
   model="gemini-2.5-flash",
   description="Agent to answer questions using Google Search.",
   instruction="You are an expert researcher. You always stick to the facts.",
   tools=[google_search]
)

# On Windows, you might neet to execute once before running this agent for the first time:
# export SSL_CERT_FILE="$(python -m certifi)"
