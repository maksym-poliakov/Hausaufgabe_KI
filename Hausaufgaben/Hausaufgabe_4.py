import time
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from requests import ReadTimeout


load_dotenv()

api_key = os.getenv("gemini_api_key")

timeout_sec = 10
client = genai.Client(api_key=api_key,http_options=types.HttpOptions(timeout=timeout_sec * 1000 ))

@retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1,min=2,max=10 ))
def get_gemini_response(prompt):

    time.sleep(0.3)
    try:
        response = client.models.generate_content(model="gemini-2.0-flash",contents=[prompt],)
        return response.text
    except ReadTimeout:
        return f"Запрос к Gemini AI превысил время ожидания {timeout_sec } сек."
    except Exception as e :
        return f"Ошибка : {str(e)}"


if __name__ == "__main__" :
    responses = get_gemini_response('какая погода сегодня в Киеве')
    print(responses)

