# API Examples

This document provides `curl` examples for interacting with the OpenAI-compatible APIs exposed by the Model Service Platform.

## Example 1: Sending a Chat Completion Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me a joke."}
    ],
    "max_tokens": 32,
    "temperature": 0.7,
    "top_p": 0.9,
    "n": 1,
    "stream": false,
    "stop": ["\nUser:", "\nAssistant:"]
  }'