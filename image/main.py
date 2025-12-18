from dotenv import load_dotenv
from openai import OpenAI
import os


load_dotenv()
client=OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response=client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role":"user",
         "content":[
             {"type":"text","text":"Genearte a caption for this image in 50 words"},
             {"type":"image_url","image_url":{"url":"https://images.pexels.com/photos/577585/pexels-photo-577585.jpeg"}}
         ]}
    ]
)

print(response.choices[0].message.content)