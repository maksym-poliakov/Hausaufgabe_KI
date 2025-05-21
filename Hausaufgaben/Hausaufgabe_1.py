from pathlib import Path
from google import genai
import dotenv
import os

dotenv.load_dotenv(Path('.env'))

api_key = os.environ.get('gemini_api_key')


client = genai.Client(api_key=api_key)

def client_response(text_response:str)  :
    response = client.models.generate_content(model="gemini-2.0-flash",contents=[text_response])

    return response.text

str_input = input('Введите текст запроса к Gemini : ')

print(client_response(str_input))