curl -X "POST" "https://mk1--mk1-chat-endpoint-dev.modal.run/generate" -H 'Content-Type: application/json' -d '{
  "text": "What is the difference between a llama and an alpaca?",
  "max_tokens": 512,
  "eos_token_ids": [1, 2],
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 1.0
}'