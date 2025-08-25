"""Locust load testing script for LLM containers"""

import random

from locust import HttpUser, task, between, tag

MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"  # Match settings.MODEL_NAME
HEADERS = {"Content-Type": "application/json"}

USER_PROMPTS = [
    "Tell me a fun fact about space.",
    "What’s the capital of Australia?",
    "Write me a short poem about the ocean.",
    "Explain quantum computing in simple terms.",
]


class ChatUser(HttpUser):
    """User class for simulating chat requests to the LLM API."""

    # Shorter wait times for quick stress; adjust for heavier runs
    wait_time = between(0.5, 2)

    @tag("light_non_streaming")
    @task(4)
    def chat_completion_quick(self):
        """Quick non-streaming request to test server response."""
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": random.choice(USER_PROMPTS)}],
            "max_tokens": 10,  # keep small for CPU test
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stream": False,
        }
        self.client.post("/v1/chat/completions", json=payload, headers=HEADERS)

    @tag("light_streaming")
    @task(2)
    def chat_completion_streaming_quick(self):
        """Quick streaming request to test server response."""
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": random.choice(USER_PROMPTS)}],
            "max_tokens": 10,  # keep small for CPU test
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stream": True,
        }
        self.client.post("/v1/chat/completions", json=payload, headers=HEADERS)

    @tag("heavy_non_streaming")
    @task(1)
    def chat_completion_heavy(self):
        """Heavier load with larger max_tokens."""
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "Give me a 200-word story about a dragon who learns to cook.",
                }
            ],
            "max_tokens": 50,  # heavier load case
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stream": False,
        }
        self.client.post("/v1/chat/completions", json=payload, headers=HEADERS)
