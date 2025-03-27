# Simple chat-like GPT-2 Model

As a way to learn fine-tuning I will try to fine-tune a base GPT-2 model on a question-answer chat-like dataset in order to create a very simple conversational chatbot.

**GPT-2** is a **Causal Language Model (CLM)**, meaning it's trained to predict the next token in a sequence. By fine-tuning it on question-answer pairs, we're teaching it to generate answers after seeing questions, mimicking a chat-like interaction.

Example training data format:

```ini
Human: Hello, how are you doing?
Bot: Hi, I am fine! How about yourself?
```

The model learns to predict "Hi, I am fine! How about yourself?" when given "Hello, how are you doing?".