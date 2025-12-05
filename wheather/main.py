from dotenv import load_dotenv
from openai import OpenAI
import os
import requests

import json
load_dotenv()
client=OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_weather(city:str):
    url=f"https://wttr.in/{city.lower()}?format=%C+%t"
    response=requests.get(url)

    if response.status_code==200:
        return f"The weather in {city} is {response.text}"
    
    return "Something went wrong"



def main():
    user_query=input("> ")
    response=client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role":"user","content":user_query}
        ]
    )

    print(f"Agent:{response.choices[0].message.content}")

# if __name__ == "__main__":
#     main()

print(get_weather("New Delhi"))
