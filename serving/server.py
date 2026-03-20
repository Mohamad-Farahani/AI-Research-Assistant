from vllm import LLM, SamplingParams

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=200
)

prompt = "Explain reinforcement learning in simple terms."

outputs = llm.generate(prompt, sampling_params)

for o in outputs:
    print(o.outputs[0].text)