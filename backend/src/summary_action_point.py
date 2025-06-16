import os

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_action_items(transcript):
    prompt = f"""
    You are an assistant that extracts action points from meeting transcripts.
    
    Transcript:
    \"\"\"
    {transcript}
    \"\"\"
    - provide summary and extract the action points in bullet format, including:
        - What needs to be done
        - Who is responsible
        - Any due dates etc
    - add who said what like speaker name and his speach
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts action points from meeting transcripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return (response.choices[0].message.content)
