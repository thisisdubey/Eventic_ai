import warnings
warnings.filterwarnings('ignore')

import os
from crewai import Agent, Crew, Task, LLM
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel
import json
import streamlit as st
from typing import Dict, Any  # For type hinting

# --- Configuration ---
def configure_environment():
    # Use environment variables or a more secure method for API keys in production
    from utils import get_serper_api_key, get_gemini_api_key, get_gemini_model_name  # Move to utils.py
    os.environ["GEMINI_API_KEY"] = get_gemini_api_key() # Get these from environment variables or a secrets manager
    os.environ["GEMINI_MODEL_NAME"] = get_gemini_model_name()
    os.environ["SERPER_API_KEY"] = get_serper_api_key()

configure_environment()

my_llm = LLM(
    model=os.environ["GEMINI_MODEL_NAME"],
    api_key=os.environ["GEMINI_API_KEY"]
)

# --- Tools ---
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# --- Pydantic Model ---
class VenueDetails(BaseModel):
    name: str = ""
    address: str = ""
    capacity: int = 0
    booking_status: str = ""


# Agent 1: Venue Coordinator
venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue "
         "based on event requirements in {event_city} of country {event_country}",
    tools=[search_tool, scrape_tool],
    llm=my_llm,
    verbose=True,
    backstory=(
        "With a keen sense of space and "
        "understanding of event logistics, "
        "you excel at finding and securing "
        "the perfect venue that fits the event's theme, "
        "size, and budget constraints."
    )
)

# Agent 2: Logistics Manager
logistics_manager = Agent(
    role='Logistics Manager',
    goal=(
        "Manage all logistics for the event in {event_city} of country {event_country}, "
        "including catering and equipment"
    ),
    tools=[search_tool, scrape_tool],
    llm=my_llm,
    verbose=True,
    backstory=(
        "Organized and detail-oriented, "
        "you ensure that every logistical aspect of the event "
        "from catering to equipment setup "
        "is flawlessly executed to create a seamless experience."
    )
)

# Agent 3: Marketing and Communications Agent
marketing_communications_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Effectively market the event in {event_city} of country {event_country} "
         "and communicate with participants",
    tools=[search_tool, scrape_tool],
    llm=my_llm,
    verbose=True,
    backstory=(
        "Creative and communicative, "
        "you craft compelling messages and "
        "engage with potential attendees "
        "to maximize event exposure and participation."
    )
)

# ## Creating Tasks
# - By using `output_json`, you can specify the structure of the output you want.
# - By using `output_file`, you can get your output in a file.
# - By setting `human_input=True`, the task will ask for human feedback (whether you like the results or not) before finalising it.


venue_task = Task(
    description="Find a venue in {event_city} of country {event_country} "
                "that meets criteria for {event_topic}.",
    expected_output="All the details of a specifically chosen"
                    "venue you found to accommodate the event.",
    human_input=False,
    output_json=VenueDetails,
    output_file="venue_details.json",
    agent=venue_coordinator
)

# - By setting `async_execution=True`, it means the task can run in parallel with the tasks which come after it.
logistics_task = Task(
    description="Coordinate catering and "
                "equipment for an event "
                "with {expected_participants} participants "
                "on {tentative_date}.",
    expected_output="Confirmation of all logistics arrangements "
                    "including catering and equipment setup.",
    human_input=False,
    async_execution=True,
    agent=logistics_manager
)

marketing_task = Task(
    description="Promote the {event_topic} "
                "aiming to engage at least"
                "{expected_participants} potential attendees.",
    expected_output="Report on marketing activities "
                    "and attendee engagement formatted as markdown.",
    async_execution=False,
    output_file="marketing_report.md",  # Outputs the report as a text file
    agent=marketing_communications_agent
)

# --- Crew ---
event_management_crew = Crew(
    agents=[venue_coordinator, logistics_manager, marketing_communications_agent],
    tasks=[venue_task, logistics_task, marketing_task],
    verbose=True  # Set to False in production
)

event_details: Dict[str, Any] = {
    'event_topic': "Tech Innovation Conference",
    'event_description': "A gathering of tech innovators "
                         "and industry leaders "
                         "to explore future technologies.",
    'event_city': "Hyderabad",
    'event_country': "India",
    'tentative_date': "2025-03-15",
    'expected_participants': 1000,
    'budget': 10000,
    'venue_type': "Conference Hall"
}

# --- Streamlit UI ---
st.title("Event Management AI")

# Use st.session_state for persistence
if "event_details" not in st.session_state:
    st.session_state.event_details = event_details  # Initialize

# Input fields (Streamlit)
for key, value in st.session_state.event_details.items():
    st.session_state.event_details[key] = st.text_input(key.replace("_", " ").title(), value)

# --- Run Crew ---
if st.button("Run"):
    with st.spinner("Running..."):
        try:
            result = event_management_crew.kickoff(inputs=st.session_state.event_details)

            # Display Outputs (Improved error handling)
            st.subheader("Venue Details")
            try:
                with open('venue_details.json', 'r') as f:  # Specify read mode 'r'
                    venue_data = json.load(f)
                st.json(venue_data)
            except (FileNotFoundError, json.JSONDecodeError) as e:  # Handle both errors
                st.error(f"Error loading venue details: {e}")

            st.subheader("Marketing Report")
            try:
                with open('marketing_report.md', 'r') as f:
                    marketing_report = f.read()
                st.markdown(marketing_report)
            except FileNotFoundError:
                st.error("Marketing report not found. It might take a few seconds to generate.")
            except Exception as e:
                st.error(f"Error loading marketing report: {e}")


        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)  # Print traceback for debugging (remove in production)