# GPT-2 Model Response

To improve the response we got and prevent the model from repeating itself, since **gpt-2** doesn't have a built-in mechanism to detect that, we can start by adding the parameter `no_repeat_ngram_size`. 

If the model is **repeating phrases**, you can **prevent n-gram repetition** with `no_repeat_ngram_size=1`. An **n-gram** is a sequence of `n` items (usually words or tokens) in a text, so size 1 means a singel token or word.
- **Unigram (1-gram)**: A single word or token, `"AI"`.
- **Bigram (2-gram)**: A pair of words, `"future of"`.
- **Trigram (3-gram)**: Three words, `"the future of"`.

GPT-2 operates on tokens, not words, meaning it predicts the next token in the sequence, not the next word, so **n-grams**, in our case, refer to `token-level n-grams`, not `word-level n-grams`.

**n-grams** help us analyze patterns in text, for example, the bigram `"future of"` might often be followed by `"AI"` in a text about technology.

GPT-2 is a **language model**, meaning it predicts the next word (or token) in a sequence based on the previous words. It does this by learning patterns in the text, including **n-grams**.

Sometimes, GPT-2 gets stuck in a loop and repeats the same n-grams over and over. For example repetitive phrases like, *"The future of AI is bright. The future of AI is bright. The future of AI is bright..."*, but instead we would like an output like, *"The future of AI is bright and full of potential."*.

This happens because the model is too confident in certain patterns and keeps repeating them.

The `no_repeat_ngram_size` parameter prevents the model from repeating n-grams of a certain size. If you set `no_repeat_ngram_size=2`, the model will **never repeat any 2-gram (bigram)** in the generated text. If you set `no_repeat_ngram_size=4`, the model will **never repeat any 4-gram** in the generated text.

When generating text, the model keeps track of the `n-grams` it has already used. If it tries to generate an `n-gram` that has already appeared, it blocks that option and chooses a different token (word) instead.

The generated text becomes more diverse and natural-sounding and you can control how strict the repetition prevention is by adjusting the `n-gram` size.

If you set `no_repeat_ngram_size` too small, the model may still allow longer repetitive patterns (3-grams or 4-grams), if set too large, it may restrict the model too much, making it harder to generate coherent text.

```python
output = model.generate(
    **input_ids, 
    max_length=100,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id   
)
```

This prevents repeating any `2-gram` (two-token sequence) and the response we get is much better.

### Probabilistic Sampling

If the model keeps looping the same words or you are not happy with the output, try making it more random.

```python
output = model.generate(
    **input_ids, 
    max_length=100,         # Generate up to 100 tokens
    no_repeat_ngram_size=2, # Prevent repeating any 2-token sequence
    do_sample=True,         # Enables probabilistic sampling
    temperature=0.8,        # Increases randomness
    top_p=0.9,              # Ensures more diverse words
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id  
)
```

Setting `do_sample=True` tells the model to **sample** the next token probabilistically instead of always choosing the most likely token (greedy decoding). Without sampling, the model always picks the token with the highest probability, which can lead to boring, repetitive or overly deterministic text. Sampling introduces randomness, making the output more creative and diverse.

With sampling set to true, the model calculates a probability distribution over all possible tokens. Instead of picking the token with the highest probability, it randomly selects a token based on the probabilities.

`temperature` controls the randomness of the sampling process, if set low (0.1) the model becomes more deterministic and conservative. It favors high-probability tokens and avoids risky choices. If set high (1.0) the model becomes more random and creative. It’s more likely to pick lower-probability tokens, leading to more diverse (but potentially less coherent) output.

A `temperature=0.5` strikes a balance between creativity and coherence, so lower temperatures are good for factual or precise text, while higher temperatures are good for creative or exploratory text.

`top_p=0.7` is called **nucleus sampling** or **top-p sampling** and it restricts sampling to the smallest set of tokens whose cumulative probability exceeds `p` (in this case, 0.7). For example, if the top tokens have probabilities `[0.5, 0.3, 0.1, 0.05, ...]`, the model will only sample from the first few tokens that add up to at least 0.7 (in this case, the first two tokens: 0.5 + 0.3 = 0.8).

Top-p sampling ensures that the model only considers plausible tokens while still introducing randomness. It avoids picking very unlikely tokens, which can lead to nonsensical output.

The model first sorts the tokens by probability, adds up the probabilities until the cumulative probability exceeds `p` and samples only from this subset of tokens.

`do_sample=True` is required, since it enables probabilistic sampling and without it, the model uses **greedy decoding** (always picking the most likely token) and parameters like `temperature` and `top_p` have no effect.

If you have seen the parameters `num_beams` and `early_stopping`, they are used together in a different decoding strategy called **beam search**, which is used to find the most probable sequence by exploring multiple possible paths.

**Beam search** is slower than **nucleus sampling**, because it expands multiple possibilities before deciding and `early_stopping=True` helps prevent overly long outputs when using `num_beams`. 

**Nucleus sampling**, which we are using with `top_p`, selects tokens based on probability, allowing diversity in responses. This is better for creative text generation and is faster since it doesn't expand multiple possibilities. So `num_beams` and `early_stopping=True` have no benefit for our setup.

Basically, **beam Search** (`num_beams`) and **nucleus sampling** (`top_p`) are two different decoding strategies and cannot be used together effectively. 

Updated code sample:

```python
output = model.generate(
    **inputs, 
    max_length=100,               # Generate up to 100 tokens
    no_repeat_ngram_size=2,       # Prevent repeating any 2-token sequence
    do_sample=True,               # Enable probabilistic sampling
    temperature=0.7,              # Balance creativity and coherence
    top_p=0.9,                    # Use nucleus sampling with p=0.9
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id  
)
```

Imagine a storyteller:
- `do_sample=True`: The storyteller doesn't always tell the same story, will add some variety.
- `temperature=0.7`: The storyteller is creative but doesn't go too far off-script.
- `top_p=0.9`: The storyteller only uses ideas that make sense in the context of the story.
- `no_repeat_ngram_size=2`: The storyteller avoids repeating the same phrases over and over.

This combination keeps the story fresh, interesting and coherent!

If you're still not happy with the response, we might need to add something more.

### Penalize Repeated Tokens

The paremeter `repetition_penalty=1.2` discourages the model from repeating the same words or phrases too often. It does this by **penalizing** the probability of tokens that have already been generated. A `repetition_penalty` greater than `1.0` reduces the likelihood of repeated tokens and a `repetition_penalty` less than `1.0` encourages repetition (though this is rarely useful).

Even with `no_repeat_ngram_size`, the model might still overuse certain words or phrases, so the repetition penalty might help to ensure the output stays diverse and avoids redundancy.

This works because the model calculates the probability of each token and if a token has already appeared in the generated text, its probability is divided by the `repetition_penalty` (1.2). The result is the repeated tokens will less likely to be chosen again.

So by setting `repetition_penalty=1.2`, the model reduces the probability of tokens that have already been used in the generated text, which prevents the model from overusing the same words or phrases.

The parameter `eos_token_id=tokenizer.eos_token_id` tells the model to stop generating text when it encounters the **end-of-sequence (EOS) token**. The EOS token is a special token that marks the end of a sequence. In GPT-2, the EOS token is usually `"<|endoftext|>"`.

Without this parameter the model might keep generating text indefinitely, even after it has logically finished its response. The EOS token acts like a "stop sign" for the model.

Updated code sample:

```python
output = model.generate(
    **inputs, 
    max_length=100,             # Generate up to 100 tokens
    no_repeat_ngram_size=2,     # Prevent repeating any 2-token sequence
    do_sample=True,             # Enable probabilistic sampling
    temperature=0.5,            # Balance creativity and coherence
    top_p=0.9,                  # Use nucleus sampling with p=0.9
    repetition_penalty=1.1,     # Penalize repeated tokens
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
```

Back to our storyteller:
- `repetition_penalty=1.2`: The storyteller avoids using the same words or phrases too often, making the story more engaging.
- `eos_token_id=tokenizer.eos_token_id`: The storyteller knows when to stop, so the story doesn't drag on unnecessarily.

This combination keeps the story interesting, concise and well-paced! Still, the response might stop in the middle of a sentence and to fix this we must tell it to stop at a period.


### Stop at Punctuation

By default, the `.generate()` method stops when it reaches `max_length=100` tokens (which may cut off a sentence mid-way) or it encounters `eos_token_id` (but not all models use EOS tokens naturally in free-text generation).

This means responses can feel unnatural because they may stop abruptly in the middle of a sentence.

We can create a **custom stopping rule** that tells the model when to stop generating.

```python
class StopAtPunctuation(StoppingCriteria):
    def __init__(self, stop_words):
        self.stop_words = stop_words

    def __call__(self, input_ids, scores, **kwargs):
        # Ensure clean text
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        suffixes = tuple(self.stop_words)

        if decoded_text.endswith(suffixes):
            return True
        return False


stopping_criteria = StoppingCriteriaList([StopAtPunctuation(stop_words=[".", "!"])])
```

`def __call__(self, input_ids, scores, **kwargs)` method is automatically called **every time** the model generates a new word. `input_ids` represents the tokens generated so far and `scores` contains model confidence scores (but we don't use it here).
   
`decoded_text = tokenizer.decode(input_ids[0])` converts all generated tokens back into human-readable text.

We want to check for multiple characters and `endswith()` accepts a tuple of suffixes, so you can either convert a list to a tuple or just use a tuple in the first place.

`if decoded_text.endswith(suffixes)` checks if the last character is a period or an exclamation mark, if `True` then the model will stop generating text, else keep generating.

We add our custom stopping rule to the `.generate()` method.

```python
output = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,   
    stopping_criteria=stopping_criteria  # Add custom stopping criteria
)
```

The model starts generating words and after every new word, the model checks if the generated text ends with a punctuation (`".", "!"`). If so, stop generating immediately, else keep going until it reaches `max_length=100`.

Now the model always finishes its sentences naturally before stopping. So, instead of stopping at a **random** length, we make sure the model **always** finishes its sentences.

This might cause the stopping criteria to kick in at the first punctuation, and short answers like *"The future of AI is bright."* might feel somewhat **underwhelming**.

To encourage longer responses, we can modify the stopping behavior, while still ensuring complete sentences, by allowing multiple sentences.

Instead of stopping at the first period, we can require at least a certain number of sentences before stopping.


```python
import re

class StopAtPunctuation(StoppingCriteria):
    def __init__(self, stop_words, min_sentences=2):
        self.stop_words = stop_words
        self.min_sentences = min_sentences

    def __call__(self, input_ids, scores, **kwargs):
        # Ensure clean text
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        suffixes = tuple(self.stop_words)

        # Convert list into a regex pattern
        regex = f"[{''.join(re.escape(word) for word in self.stop_words)}]+"
        matches = re.findall(regex, decoded_text)

        # Count the number of complete sentences
        sentence_count = len(matches)

        if sentence_count >= self.min_sentences and decoded_text.endswith(suffixes):
            return True
        return False
    
# Require at least 3 sentences before stopping
stopping_criteria = StoppingCriteriaList(
    [StopAtPunctuation(stop_words=[".", "!"], min_sentences=3)]
)
```

The model must generate at least 3 sentences before stopping at a punctuation, this allows longer responses while still ensuring grammatical completeness.

Use `repetition_penalty` and `top_p` for more substance, because sometimes, short responses come from the model **playing it safe**. You can force it to be more diverse and add more depth with these tweaks:

```python
output = model.generate(
    **inputs,
    max_length=512,
    do_sample=True,
    temperature=0.8,          # Encourage more creativity
    top_p=0.95,               # Allow slightly riskier choices
    repetition_penalty=1.05,  # Avoid repeating simple phrases
    stopping_criteria=stopping_criteria  
)
```

`temperature=0.8` will generate more variety in responses, `top_p=0.95` make sure to include slightly riskier but interesting words and `repetition_penalty=1.05` prevents repetitive phrasing.  

Now, instead of *"The future of AI is bright."*, the model might say, *"The future of AI is bright. As artificial intelligence advances, we will see groundbreaking innovations in healthcare, automation, and even creative fields. However, ethical considerations remain a challenge."*  

Now your AI **won’t** give one-liner responses but also won't ramble forever!