---
base_model: Qwen/Qwen2.5-32B
language:
- en
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/LICENSE
pipeline_tag: text-generation
tags:
- chat
- mlx
---

# mlx-community/Qwen2.5-32B-Instruct-bf16

The Model [mlx-community/Qwen2.5-32B-Instruct-bf16](https://huggingface.co/mlx-community/Qwen2.5-32B-Instruct-bf16) was converted to MLX format from [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) using mlx-lm version **0.18.1**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-32B-Instruct-bf16")
response = generate(model, tokenizer, prompt="hello", verbose=True)
```
