# Simple chat-like GPT-2 Model

As a way to learn fine-tuning I will try to fine-tune a base GPT-2 model on a question-answer dataset in order to create a simple chat-like model.

**GPT-2** is a **Causal Language Model (CLM)**, meaning it's trained to predict the next token in a sequence. By fine-tuning it on question-answer pairs, we're teaching it to generate answers after seeing questions, mimicking a chat-like interaction.

Example training data format:

```ini
User: What is the capital of France? 
Assistant: The capital of France is Paris.
```

The model learns to predict "The capital of France is Paris." when given "What is the capital of France?".