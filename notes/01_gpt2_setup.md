# GPT-2 Setup

Start by installing the `transformers` and `torch` libraries, which we need to use pre-trained models like GPT-2.

```bash
pip install transformers torch
```

From transformers, import two classes:
- `GPT2Tokenizer`: This is the tokenizer for GPT-2. It converts text into tokens (numbers) that the model can understand.
- `GPT2LMHeadModel`: This is the GPT-2 model itself, specifically the version with a "language modeling head" (LMHead), which is used for generating text.

The **tokenizer** prepares the input for the model and the **model** does the heavy lifting of generating text.

I'm going to use Hugging Face’s auto-classes, which are designed to automatically detect and load the correct model and tokenizer based on the model name or path you provide.
- `AutoTokenizer`: Automatically selects the correct tokenizer for the model.
- `AutoModelForCausalLM`: Automatically selects the correct model class for causal language modeling (like GPT-2).

`AutoModelForCausalLM` or `GPT2LMHeadModel` requires the **PyTorch** library (`torch`). It's a python deep learning framework and essential for working with models like GPT-2. 

**PyTorch** provides tools for building, training and running neural networks. Think of it as a toolbox for working with machine learning models.

When you use `AutoModelForCausalLM` or `GPT2LMHeadModel`, you're working with a **neural network** (GPT-2). Neural networks are complex mathematical systems that require:
- **Tensors**: Multi-dimensional arrays (like vectors or matrices) to represent data.
- **Automatic Differentiation**: A way to calculate gradients (used for training models).
- **GPU Acceleration**: The ability to run computations on GPUs for faster processing.

PyTorch provides all of these tools. Without it, you wouldn't be able to:
- Load the model's weights.
- Perform computations (like generating text).
- Train or fine-tune the model.

For example, when loading the model, `GPT2LMHeadModel.from_pretrained("gpt2")`, it loads the GPT-2 model's architecture and weights into memory and the model’s weights are stored as **tensors** (multi-dimensional arrays). PyTorch handles these tensors and ensures they're ready for computation.

Imagine you’re building a car:
- **PyTorch** is like the factory that provides the tools and machinery to assemble the car.
- **Tensors** are the raw materials (metal, plastic, etc.) that the factory uses.
- **Automatic differentiation** is the quality control system that ensures the car is built correctly.
- **GPU acceleration** is like using a high-speed assembly line to build the car faster.

Without PyTorch, you’d have to build everything from scratch, like trying to assemble a car without a factory or tools!

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

Deep learning models (like GPT-2) perform millions of calculations. GPUs (Graphics Processing Units) are specialized hardware for performing many computations in parallel and they are much faster at these calculations than CPUs, so we want to use them if available. 

`torch.device` is a PyTorch object that specifies where tensors and models should live, on the **CPU** or **GPU**. 

```python
# Check for GPU and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

When you load a model using `from_pretrained`, Hugging Face automatically places the model on the CPU by default.

If you want to move the model to the GPU, you need to do it explicitly. Similarly, input tensors need to be moved to the same device as the model (done later).

```python
# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Move the model to the device
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
```
`tokenizer = GPT2Tokenizer.from_pretrained("gpt2")` loads the GPT-2 tokenizer, which is used to break text into tokens and convert tokens back into text.

`model = GPT2LMHeadModel.from_pretrained("gpt2")` loads the GPT-2 model with its pre-trained weights, menaing it’s ready to generate text.

The **tokenizer** and **model** are like a translator and a writer. The translator (tokenizer) converts your words into a language the writer (model) understands and the writer generates new text based on what it's learned.

Time for the **input**, which is the text prompt you want the model to complete. It’s like giving the model a starting sentence.

```python
prompt = "The future of Artificial Intelligence"
```

GPT-2 is a **Language Model**, meaning it predicts what comes next in a sequence. You give it a starting point and it generates the rest.

We must set a **Padding Token**, which is a special token used to fill in the extra spaces for shorter sequences to match the length of the longest sequence in a batch. This is necessary because many machine learning models require input data of a consistent size.

GPT-2 does not have a default padding token, because it was originally trained without one. However, when we batch-process text (like padding/truncating to the same length), we need one.

To fix this, we can use the eos_token `<|endoftext|>` as padding, since GPT-2 already knows `<|endoftext|>`, so we reuse it as pad_token.

```python
tokenizer.pad_token = tokenizer.eos_token
```

This ensures all input sequences are the same length. By setting `pad_token` to `eos_token`, we're telling the tokenizer to use the same token for both padding and marking the end of a sequence.

Time to tokenize the input, `tokenizer(prompt)` converts the text prompt into tokens (numbers). For example, "The future of AI " might become `[464, 1234, 567, 7890, 345]`.

```python
# Move input tensors to the device
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
```

- `return_tensors="pt"` tells the tokenizer to return the tokens as **PyTorch tensors** (`pt` stands for PyTorch). Tensors are like multi-dimensional arrays and PyTorch is the framework used to handle them.
- `padding=True` ensures the input is padded to a fixed length (required for batching, though here we're only processing one input).

The model expects numerical input in the form of tensors. The tokenizer converts text into these tensors, and padding ensures consistency.

To generate a response use the `generate(**inputs)` method, which tells the model to generate text based on the tokenized input. The `**inputs` unpacks the dictionary returned by the tokenizer into keyword arguments

```python
output = model.generate(**inputs)
```

The model now predicts the next token in the sequence repeatedly until it reaches a stopping condition (e.g., max length or end-of-sequence token).

This is where the magic happens! The model uses its knowledge (learned during training) to generate a coherent continuation of the input prompt. 

Finally decode the output and print the response, `tokenizer.decode(...)` converts the tokens back into human-readable text.

```python
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)
```

The `generate` method returns a tensor of shape `(batch_size, sequence_length)`. Since we're only generating one sequence, we take the first element (`output[0]`) to get the generated tokens.

Tensors are multi-dimensional arrays. In this case, the tokenizer returns a tensor of shape `(1, sequence_length)` because we're processing one input sequence.

For example, if the generated tokens are `[[464, 1234, 567, 7890, 345, 678]]`, `output[0]` gives `[464, 1234, 567, 7890, 345, 678]`.

`skip_special_tokens=True` removes special tokens like `eos_token` from the final output, which is needed because the model generates tokens (numbers), but we want text. The tokenizer decodes these tokens back into words.

Imagine GPT-2 as a super-smart parrot. You give it a sentence ("The future of Artificial Intelligence") and it tries to complete it based on patterns it's seen before. The tokenizer is like a translator that turns your words into a secret code the parrot understands. The parrot (model) then squawks back in code and the translator turns that back into words for you to understand.

The tensor (`return_tensors="pt"`) is just a fancy way of saying "give me the code in a format I can work with" and `output[0]` is like picking the first (and only) squawk from the parrot's response.

```ini
The future of Artificial Intelligence is a very exciting one. It is a very exciting time for AI.  It is
``` 

Now this response was not that awesome, so let's do something about it!