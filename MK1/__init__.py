import modal

Model = modal.Cls.lookup(
    "mk1-flywheel-latest-llama2-7b-chat", "Model", workspace="mk1"
).with_options(
    gpu=modal.gpu.A10G(),
)

model = Model()
prompt = "[INST] What is the difference between a llama and an alpaca? [/INST] "

print(f"Prompt:\n{prompt}\n")

responses = model.generate.remote(text=prompt, max_tokens=512, eos_token_ids=[1, 2])
response = responses["responses"][0]["text"]

print(f"Response:\n{response}")
