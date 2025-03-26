# GPT-2 Model Setup

In order to use a language model like gpt-2, start by installing the `transformers`, `torch` and `accelerate` libraries. Transformers let you access thousands of pretrained models from **Hugging Face** and accelerate is a powerful tool from **Hugging Face** designed to optimize memory usage and computation across different hardware setups, like single/multi-GPU and CPU.

Start by activating our virtual environment and install the necessary packages.

```bash
# Activate the venv
source .venv/bin/activate

# On windows
.venv\Scripts\activate

# Install packages
pip install transformers torch accelerate
```

From transformers, import two classes:
- `GPT2Tokenizer`: This is the tokenizer for gpt-2, which converts text into tokens (numbers) that the model can understand.
- `GPT2LMHeadModel`: This is the actual gpt-2 model, specifically the version with a "language modeling head" (LMHead), which is used for generating text.

The **tokenizer** prepares the input for the model and the **model** does the heavy lifting of generating text.

Instead of the specific classes, we're going to use Hugging Face's auto-classes, which are designed to automatically detect and load the correct model and tokenizer based on the model name or path you provide.
- `AutoTokenizer`: Automatically selects the correct tokenizer for the model.
- `AutoModelForCausalLM`: Automatically selects the correct model class for causal language modeling, like gpt-2.

Whether you use `AutoModelForCausalLM` or `GPT2LMHeadModel`, you'll need **PyTorch** (`torch`), which is a python deep learning framework and essential for working with models like gpt-2. If you don't you'll get an **ImportError** saying `AutoModelForCausalLM` requires the PyTorch library.

**PyTorch** provides tools for building, training and running neural networks. Think of it as a toolbox for working with machine learning models.

When you use `AutoModelForCausalLM` or `GPT2LMHeadModel`, you're working with a **neural network** (gpt-2) and neural networks are complex mathematical systems that require:
- **Tensors**: Multi-dimensional arrays (like vectors or matrices) to represent data.
- **Automatic Differentiation**: A way to calculate gradients (used for training models).
- **GPU Acceleration**: The ability to run computations on GPUs for faster processing.

PyTorch provides all of these tools, so without it, you wouldn't be able to:
- Load the model's weights.
- Perform computations (like generating text).
- Train or fine-tune the model.

For example, when loading the model, `GPT2LMHeadModel.from_pretrained("gpt2")`, it loads the small 117M parameters gpt-2 model's architecture and weights into memory and the model’s weights are stored as **tensors** (multi-dimensional arrays). PyTorch handles these tensors and ensures they're ready for computation.

You can pick one of these OpenAI GPT-2 English models (or some model you like):
- `gpt2`: 12-layer, 768-hidden, 12-heads and 117M parameters.
- `gpt2-medium`: 24-layer, 1024-hidden, 16-heads and 345M parameters.
- `gpt2-large`: 36-layer, 1280-hidden, 20-heads and 774M parameters.
- `gpt2-xl`: 48-layer, 1600-hidden, 25-heads and 1558M parameters.

Imagine you’re building a car, **PyTorch** is like the factory that provides the tools and machinery to assemble the car.
- **Tensors** are the raw materials (metal, plastic, etc.) that the factory uses.
- **Automatic differentiation** is the quality control system that ensures the car is built correctly.
- **GPU acceleration** is like using a high-speed assembly line to build the car faster.

Without PyTorch, you’d have to build everything from scratch, like trying to assemble a car without a factory or tools!

### The Tokenizer and the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

Deep learning models, like gpt-2, perform millions of calculations and the **GPU** or *Graphics Processing Unit*, is specialized hardware for performing many computations in parallel, meaning it's much faster at these calculations than a **CPU** or *Central Processing Unit*. 

So if you have a **GPU** that supports **CUDA**, you want to use it. `torch.device` is a PyTorch object that specifies where tensors and models should live, on the **CPU** or on the **GPU**. 

**CUDA** (*Compute Unified Device Architecture* ) is a parallel computing framework created by **NVIDIA** that allows developers to use **GPU**s for general-purpose computing (beyond just graphics). By using **CUDA** it's possible to use NVIDIA GPUs for deep learning by writing parallel code in python (via libraries like **PyTorch** and **TensorFlow**).

```python
# Check for GPU and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

We're using Hugging Face's `.from_pretrained()` to load our model which automatically places the model on the CPU by default.

If you want to move the model to the **GPU**, you need to do it explicitly. Similarly, **input tensors** need to be moved to the same device as the model (this is done later).

```python
# Move the model to the device
model = AutoModelForCausalLM.from_pretrained(
    "gpt2"
).to(device)

# Tokenize the input and move to the device
inputs = tokenizer(
    prompt, 
    return_tensors="pt"
).to(device)
``` 

Our setup is going to look like this:

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set eos_token as pad_token for the tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Move the model to the device
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", 
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

- `tokenizer = AutoTokenizer.from_pretrained("gpt2")` loads the gpt-2 tokenizer, which is used to break text into tokens and convert tokens back into text.
- `tokenizer.pad_token = tokenizer.eos_token` is necessary for gpt-2 because gpt-2 does not have a pad token by default and some functions (like loss masking during training) rely on `pad_token_id`.
- `tokenizer.padding_side="right"` is optimal for causal language modeling (ensures padding doesn’t interfere with attention masks).
- `model = AutoModelForCausalLM.from_pretrained("gpt2")` loads the GPT-2 model with its pre-trained weights, menaing it’s ready to generate text.
- `pad_token_id=tokenizer.eos_token_id` explicitly tells the model that the **pad token ID** is the same as the **EOS token ID**.
- `torch_dtype=torch.float16` force FP16 which cuts memory transfer time in half.
- `device_map="auto"` efficiently distributes the model across available devices (GPU/CPU) which is used instead of `.from_pretrained("gpt2").to(device)`.

If you use `device_map` it requires accelerate, which optimizes memory usage (so we don't run out of VRAM). It dynamically manages model sharding (splitting layers across devices), enables mixed-precision training (fp16, bf16) and gradient checkpointing (trade compute for memory) and supports offloading (moving unused layers to CPU to save VRAM).

`FP32` or float32 is the default precision (32-bit floating point), meaning high precision, but computationally expensive. `FP16` or float16s cuts the precision in half (16-bit), meaning lower precision, but *2x smaller memory footprint* (weights/activations use half the RAM/VRAM) and *2-4x faster computation* (modern GPUs have optimized FP16 pipelines).

Using `FP16` is one of the easiest ways to speed up inference without sacrificing quality, it's like a free lunch for inference, almost no quality loss, but significant speedup. Works with all modern **GPU**s (but not **CPU**), so always use for GPU inference combined with `device_map="auto"` for automatic GPU offloading.

The **tokenizer** and **model** are like a translator and a writer. The translator (tokenizer) converts your words into a language the writer (model) understands and the writer generates new text based on what it's learned.

> You don't strictly need to set `pad_token_id=tokenizer.eos_token_id` in `.from_pretrained()` if you've already set `tokenizer.pad_token = tokenizer.eos_token` before loading the model, but it's still good practice to do so.

### The Input

Time for the **input**, which is the text prompt you want the model to complete. It's like giving the model a starting sentence.

Our model is going to predict what comes next in a sequence. We give it a starting point and it generates the rest.

The reason we assign the **eos token** as the **pad token** for the tokenizer is to ensure that when padding is needed, like in batch processing, the tokenizer uses `<|endoftext|>` instead of leaving it undefined.

The **padding token** is a special token used to fill in the extra spaces for shorter sequences to match the length of the longest sequence in a batch. This is necessary because many machine learning models require input data of a consistent size.

Since gpt-2 does not have a default padding token, because it was originally trained without one, when we batch-process text (like padding/truncating to the same length), we simply need one.

The magic trick is to reuse the **eos_token** `<|endoftext|>` as **pad_token**, since gpt-2 already knows `<|endoftext|>`.

This ensures all input sequences are the same length and by setting `pad_token` to `eos_token`, we're telling the tokenizer to use the same token for both padding and marking the end of a sequence.

The tokenizer and model are **separate objects**. Setting `tokenizer.pad_token` updates the tokenizer's behavior, but the model might still need explicit guidance about what ID to use for padding during forward passes, for example in `generate()`.

By setting both the tokenizer and the model you ensure consistency across all components (tokenizer, model and generation).

If `pad_token` isn't set in the **tokenizer**, it may throw an error when processing inputs with padding.

If `pad_token_id` isn't set in the **model**, it might misinterpret padding during training or inference. 

Time to tokenize the input. `tokenizer(prompt)` converts the text prompt into tokens (a list of numbers) and returns a dictionary containing the `input_ids` (a list of tokens) and the `attention_mask` (a list of 1s and 0s indicates if a tokens is valid or should be ignored).

```python
batch = tokenizer("Hello World") 

# batch: {'input_ids': [15496, 2159], 'attention_mask': [1, 1]}

{
    'input_ids': [ token1, token2 ],
    'attention_mask': [ 1, 1 ]  # 1=real token, 0=padding
}

batch = tokenizer(["Hello", "Hello World"]) 

{
    'input_ids': [ [15496], [15496, 2159] ], 
    'attention_mask': [ [1], [1, 1] ]
}
``` 

The model expects the numerical input in the form of tensors (multi-dimensional arrays), so we need to add the parameter `return_tensors` to tell the tokenizer to convert text into these tensors. For a single input that's enough, but for a batch we must add padding to ensure consistency, by setting `padding=True` we get batched tensors with the same length.

```python
batch = tokenizer(
    ["Hello", "Hello World"], 
    return_tensors="pt", 
    padding=True,
    truncation=True
)

{
    'input_ids': tensor([[15496, 50256], [15496,  2159]]), 
    'attention_mask': tensor([[1, 0], [1, 1]])
}
``` 

- `return_tensors="pt"` tells the tokenizer to return the tokens as **PyTorch tensors** (`pt` stands for PyTorch). Tensors are like multi-dimensional arrays and PyTorch is the framework used to handle them.
- `padding=True` ensures the input is padded to a fixed length (required for batching).
- `truncation=True` cuts sequences longer than the model's maximum context window, like gpt-2's 1024 tokens.

The first input is a single word, but `[15496]` turned into `[15496, 50256]` and the attention_mask is now `[1, 0]` (0=padding), so `50256` must be the padding token. Let's see:

```python
print(tokenizer.decode([50256]))
# <|endoftext|>
print(tokenizer.pad_token)
# <|endoftext|>
``` 

 For example, `tokenizer.encode("<|endoftext|>")` will return just `[50256]`, meaning it only returns the `input_ids` or *input_ids as a tensor* if you add `return_tensors="pt"`, but it will not return the `attention_mask`. This could be useful when you need raw token IDs quickly for debugging or single sequences.

```python
# Move input tensors to the device
inputs = tokenizer(
    prompt, 
    return_tensors="pt", 
    padding=True,
    truncation=True
).to(device)
```

### The Response

To generate a response, we use the `generate()` method, which tells the model to generate text based on the tokenized input. We know that the `tokenizer(prompt)` returns a dictionary, so we must unpack that dictionary returned by the tokenizer into keyword arguments.

We can unpack a dictionary and pass its key-value pairs as arguments to a function using the `**` operator.

```python
output = model.generate(
    **inputs,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id 
)
```

It's a good practice to explicitly set both `pad_token_id` and `eos_token_id` in the `.generate()` method, even if you've already set them when loading the model, since they are separate objects.

If you manually set `tokenizer.pad_token = tokenizer.eos_token`, this setting might not always persist when calling `.generate()`, so explicitly defining it in `.generate()` ensures it's used correctly.

The model now predicts the next token in the sequence repeatedly until it reaches a stopping condition, max length or end-of-sequence token.

This is really where the magic happens! The model uses its knowledge (learned during training) to generate a coherent continuation of the input prompt. 

Finally we decode the output and print the response. `tokenizer.decode(...)` converts the tokens back into human-readable text.

```python
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)
```

- `skip_special_tokens=True` removes special tokens like `eos_token` from the final output, since we don't want them when the tokenizer decodes the tokens back into words.

The `generate()` method returns a tensor of shape `(batch_size, sequence_length)`. Since we're only generating one sequence, the tokenizer returns a tensor of shape `(1, sequence_length)`, so we take the first element (`output[0]`) to get the generated tokens.

```python
output.size()
# torch.Size([1, 55])
```

Imagine gpt-2 as a super-smart parrot. You give it a sentence, like *"The future of AI is"* and it tries to complete that sentence based on patterns it's seen before. The tokenizer is like a translator that turns your words into a secret code the parrot understands. The parrot (model) then squawks back and the translator turns the squawks into words for you to understand.

The tensor (`return_tensors="pt"`) is just a fancy way of saying "give me the code in a format I can work with" and `output[0]` is like picking the first (and only) squawk from the parrot's response.

```ini
The future of AI is in the hands of the next generation of AI.

The future of AI is in the hands of the next generation of AI.

The future of AI is in the hands of the next generation of AI.

The future of AI
``` 

Now this response was not that awesome, but let's see if we can improve it.