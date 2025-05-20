import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Retrieve the API key securely
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

if not "GROQ_API_KEY":
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

# Initialize the Groq client
groq_client = Groq(api_key="GROQ_API_KEY")
