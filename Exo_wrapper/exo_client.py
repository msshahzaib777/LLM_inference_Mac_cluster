import requests
import json

class ExoClient:
    def __init__(self, api_url, api_key=None, model="gpt-3.5-turbo"):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def chat(self, messages, temperature=0.7, max_tokens=512):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            headers=self.headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def chat_stream(self, messages, temperature=0.7):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            headers=self.headers,
            data=json.dumps(payload),
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    if line.strip() == "data: [DONE]":
                        break
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta