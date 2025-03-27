# Dataset

Instead of trying to control and force the response in a certain way, let's try to teach the model to respond the way we want. In order to do just that we must find data we can use for that purpose. 

In order to get a more chat-like feel, let's try to fine-tune the base gpt-2 model on a question-answer dataset, which is also a fantastic way to learn how fine-tuning works!

If you don't have a dataset, you can use a any type of text document structured like this: 

```
Hi, how are you doing?	I'm fine. How about yourself?
I'm fine. How about yourself?   I'm pretty good. Thanks for asking. 
```

Before loading the text document, you need to preprocess the data into a format the model can understand. GPT-2 is typically trained on sequences of text, so we need to structure our data accordingly, in other words we convert the document into a dataset suitable for fine-tuning!

We are going to use Hugging Face's **datasets** library, which is a lightweight, fast and scalable tool for working with datasets in machine learning. 
- Easy Access to thousands of NLP, vision and audio datasets via the Hugging Face Hub.  
- Efficient Processing for fast loading and memory mapping.  
- Handles large datasets without needing to download everything at once.  
- Works seamlessly with PyTorch and TensorFlow.  
- Built-in preprocessing which supports filtering, tokenization and transformations.  

```bash
pip install datasets
``` 

In our example the data appears to be a conversation in a tab-separated format, where each line contains a pair of utterances. Adding explicit speaker labels, like `Human:` and `Bot:` will significantly improve our model performance in conversational tasks and provides a clear role identification.

In the original format the model sees two text segments separated by a tab, but has no inherent way of distinguishing which one belongs to the user and which one belongs to the AI.

In the format below, the model explicitly learns `Human:` means an incoming query and `Bot:` means an expected response. This helps it generalize responses correctly when deployed.

```
Human: Hi, how are you doing?   Bot: I'm fine. How about yourself?
Human: I'm pretty good. Thanks for asking.  Bot: No problem. So how have you been?
```

This structure provides a back-and-forth dialogue, which can be used to fine-tune gpt-2 for conversational tasks. Which is exactly what we want.

```python
text_dataset = []

with open("./dialogs.txt", "r") as fh, 
    for line in fh:
        # Since it's tab-separated split there
        q, a = line.split("\t")
        # Add speaker labels
        text = f"Human: {q} Bot: {a}"
        # Append the line
        text_dataset.append(text)
```

Here's what's going to happen during training:
- You feed the model a sequence like: `Human: Hi, how are you doing? Bot: I'm fine. how about yourself?`
- The model processes the sequence token by token and tries to predict the next token at each step.
- The loss (error) is calculated based on how well the model predicts the next token.
- The model updates its weights to minimize the loss.

At no point do you mask the question. The model learns to generate the response **conditioned on the question**.

During inference (when you use the model):
- You provide a partial sequence (the question): `Human: Hi, how are you doing? Bot:`
- The model generates the next tokens to complete the sequence: `I'm fine. how about yourself?`

We should split the data into two sets, a **training set** and a **validation set** to monitor overfitting. But how should we format the validation set? Should we mask the question part?

The **validation set** should **not** be the exact same as the **training set**. Instead, it should follow the **same format**, like `Human: <question> Bot: <response>`, but the questions and answers should be **different**.

The purpose of the validation set is to evaluate how well your model generalizes to **unseen data**. If the validation set is the same as the training set:
- The model might simply memorize the training data (overfitting) and perform well on the validation set, even though it hasn’t learned to generalize.
- You won't get an accurate measure of how well the model will perform in real-world scenarios.

By using a **separate validation set** with different questions and answers, you can:
- Test the model's ability to handle new, unseen inputs.
- Detect overfitting (if the model performs well on the training set but poorly on the validation set).

The reason we don't need to mask the question during training is gpt-2 is an **autoregressive language model**, meaning it predicts the next token in a sequence based on all the previous tokens. During fine-tuning, you provide the model with a sequence of text, like `Human: Hi, how are you doing? Bot: I'm fine. how about yourself?`, and the model learns to predict the next token at every step.

The key idea is that the model learns to generate the **entire sequence**, but during inference (when you use the model), you provide a **partial sequence**, like `Human: Hi, how are you doing? Bot:`, and let the model generate the rest, `I'm fine. how about yourself?`.

The model learns the relationship between the question and the response by seeing the **full sequence**. The model is trained to predict the next token at every step, so it learns to generate the response **conditioned on the question**. Masking the question would break the sequence and prevent the model from learning the context!

For example:
- If you provide the sequence: `Human: Hi, how are you doing? Bot: I'm fine. how about yourself?`
- The model learns to predict `I'm fine. how about yourself?` given the context `Human: Hi, how are you doing? Bot:`.

Our dataset is structured exactly like that, as a dialogue, which s is a good format for training because it provides the full context (the human's question) and the expected response (the bot's reply) and the model learns to predict the bot's response based on the human's input.

### Split The Dataset

When splitting the dataset into **training** and **validation** sets, both sets should follow the same format, but should **not** be the same.

The validation set should include the full dialogue (human input + bot response) so that the training can evaluate the model's ability to generate responses given the context, meaning it should **not** contain only the bot's responses. 

Masking the input (question) during validation doesn't make sense, because if you only provided the response in the validation set, you wouldn't have a way to evaluate the model's ability to generate responses based on questions.

To create a validation set, you can split your original dataset into two parts:
- Training set with 80% of the data
- Validation set with 20% of the data

```python
train_fraction = 0.8

split_index = int(train_fraction * len(text_dataset))

train_text = text_dataset[:split_index]
val_text = text_dataset[split_index:]

# Save datasets as text files
with open("train_text.txt", "w") as fh:
    fh.writelines(train_text)

with open("val_text.txt", "w") as fh:
    fh.writelines(val_text)


train_dataset = TextDataset(
    tokenizer=tokenizer, 
    file_path="train_text.txt", 
    block_size=128
)

val_dataset = TextDataset(
    tokenizer=tokenizer, 
    file_path="val_text.txt", 
    block_size=128
)
``` 

Here we did it manually and used `TextDataset`, which is an older approach and it works, but the Hugging Face's datasets `load_dataset()` is generally preferred for fine-tuning tasks. `TextDataset` splits the text into fixed-size chunks (block_size=128) without preserving sentence boundaries, it does not include padding (each block is exactly block_size=128) and returns a PyTorch dataset (`torch.utils.data.Dataset`), which does not have the structured dataset features of `datasets.Dataset`.

So we will use `load_dataset()` and load the dataset first, which is generally better, then split the dataset using `train_test_split()` (from the datasets library) rather than splitting it manually before loding.
- More Efficient – Avoids unnecessary file I/O (writing and reading two separate text files).
- More Flexible – Easily adjust the split ratio (test_size=0.2) without rewriting files.
- Reproducibility – You can set a seed for consistent splits across runs.
- Better Integration – Works seamlessly with Hugging Face's datasets framework.

The `train_test_split` from `sklearn.model_selection` is not the same, even though it serve a similar purpose it work differently, but if your data is still in a raw format (list, array, pandas DataFrame), use you should use `sklearn.model_selection.train_test_split()`.

For our case (fine-tuning gpt-2 with datasets), we've already loaded our data into a `datasets.Dataset` with the `load_dataset` method so therefor we should stick with `datasets.train_test_split()` since it's optimized for Hugging Face workflows.

So we start by saving our dataset as a text file.

```python
with open("./text_dataset.txt", "w") as fh:
    fh.writelines(text_dataset)
``` 

From Hugging Face's `datasets` library we'll use `load_dataset("text", data_files="text_dataset.txt")` to load our text file as a Hugging Face Dataset object. By default, this creates a DatasetDict with a single key `"train"` containing all your data.


```python
dataset = {
    "train": Dataset([entire text document])
}
```

`dataset["train"].train_test_split(test_size=0.2, seed=42)` is splitting the data:
- `dataset["train"]` accesses the training portion (which is currently all our data).
- `.train_test_split()` is a Dataset method that splits this into train/test sets.
- `test_size=0.2` means 20% goes to test, 80% remains in train.
- `seed=42` ensures reproducibility.

To understand `seed` imagine you're shuffling cards, even if you try to shuffle "the same way," tiny differences (your hand movement, air resistance) make it unpredictable.  A seed is like a magic shuffle that forces randomness to pretend to be random, but actually follow a secret pattern.

If you shuffle with `seed=42`, you’ll always get the exact same "random" order every time, if you shuffle with `seed=43` you will get a different "random" order.

```python
# First run (seed=42)
split = dataset.train_test_split(seed=42)  
# Output: Train = [A, C, D], Test = [B, E]  

# Second run (same seed!)
split = dataset.train_test_split(seed=42)  
# Output: Train = [A, C, D], Test = [B, E]  # Same as before!

# Different seed (seed=99)
split = dataset.train_test_split(seed=99)  
# Output: Train = [B, D, A], Test = [C, E]  # New split!
``` 

We care about **reproducibility** because it will give us the same results tomorrow, or on your friend’s computer. Everyone who is using `seed=42` gets identical splits which means fair comparisons.

```python
# Load full dataset first
dataset = load_dataset("text", data_files="text_dataset.txt")

# Split into training and validation sets (80/20)
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

# Access train and validation sets
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# Save the splits (optional)
train_dataset.to_json("train_dataset.json", orient='records', lines=True)
val_dataset.to_json("val_dataset.json", orient='records', lines=True)

# Load both splits at once
reloaded_dataset = load_dataset('json', data_files={
    'train': 'train_dataset.json',
    'test': 'val_dataset.json'
})
``` 

Once we have the training and validation sets, we can use them to fine-tune our model!