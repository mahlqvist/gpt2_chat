# Model Response

The response we got was not that great, this repetition issue happens because **GPT-2** doesn't have a built-in mechanism to detect when it's repeating itself. 

To fix this we can start by adding the parameter `no_repeat_ngram_size`. If the model is **repeating phrases**, you can **prevent n-gram repetition** with:

An **n-gram** is a sequence of `n` items (usually words or tokens) in a text.
- **Unigram (1-gram)**: A single word, `"AI"`.
- **Bigram (2-gram)**: A pair of words, `"future of"`.
- **Trigram (3-gram)**: Three words, `"the future of"`.

GPT-2 operates on tokens, not words, meaning it predicts the next token in the sequence, not the next word, so **n-grams** refer to `token-level n-grams`, not `word-level n-grams`.

**n-grams** help us analyze patterns in text, for example, the bigram `"future of"` might often be followed by `"AI"` in a text about technology.

GPT-2 is a **language model**, meaning it predicts the next word (or token) in a sequence based on the previous words. It does this by learning patterns in the text, including **n-grams**.

Sometimes, GPT-2 gets stuck in a loop and repeats the same n-grams over and over. For example repetitive phrases like, *"The future of AI is bright. The future of AI is bright. The future of AI is bright..."*, but instead we would like an output like, *"The future of AI is bright and full of potential."*.

This happens because the model is too confident in certain patterns and keeps repeating them.

The `no_repeat_ngram_size` parameter prevents the model from repeating n-grams of a certain size. If you set `no_repeat_ngram_size=2`, the model will **never repeat any 2-gram (bigram)** in the generated text. If you set `no_repeat_ngram_size=4`, the model will **never repeat any 4-gram** in the generated text.

While generating text, the model keeps track of the n-grams it has already used. If it tries to generate an n-gram that has already appeared, it blocks that option and chooses a different word instead.

The generated text becomes more diverse and natural-sounding and you can control how strict the repetition prevention is by adjusting the n-gram size.

If you set `no_repeat_ngram_size` too small, the model may still allow longer repetitive patterns (3-grams or 4-grams), if set too large, it may restrict the model too much, making it harder to generate coherent text.

```python
output = model.generate(
    input_ids=input_ids, 
    max_length=100,
    no_repeat_ngram_size=2  
)
```

This prevents repeating any 2-gram (two-token sequence) and the response we get is much better.

```ini
The future of Articial Intelligence is in the hands of the people.
The people are the ones who are going to decide what is best for the future. 
They are not the only ones. The people have the power to change the world. 
And they are doing it by the millions.
``` 

### Probabilistic Sampling

If the model keeps looping the same words or you are not happy with the output, try making it more random.

```python
output = model.generate(
    input_ids=input_ids, 
    max_length=100,
    no_repeat_ngram_size=2,
    do_sample=True,  # Enables probabilistic sampling
    temperature=0.8, # Increases randomness
    top_p=0.9,       # Ensures more diverse words
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

Updated code sample:

```python
output = model.generate(
    **inputs, 
    max_length=100,               # Generate up to 100 tokens
    no_repeat_ngram_size=2,       # Prevent repeating any 2-token sequence
    do_sample=True,               # Enable probabilistic sampling
    temperature=0.7,              # Balance creativity and coherence
    top_p=0.9,                    # Use nucleus sampling with p=0.9
)
```

Imagine a storyteller:
- `do_sample=True`: The storyteller doesn't always tell the same story, will add some variety.
- `temperature=0.7`: The storyteller is creative but doesn't go too far off-script.
- `top_p=0.9`: The storyteller only uses ideas that make sense in the context of the story.
- `no_repeat_ngram_size=2`: The storyteller avoids repeating the same phrases over and over.

This combination keeps the story fresh, interesting and coherent!

```txt
The future of Articial Intelligence is at stake.

The next time you see a company trying to get into the art world, 
you may be wondering what's going on with their management. 
I'm sure they're all in the same boat as their investors. 
It's just that they don't know how to do it. The future is always in their hands. 
They have a lot of other things to think about. 
For instance, they can get a few years of a good deal on their
```

This response actually got worse, so with these new parameters we might need to add something.

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
    eos_token_id=tokenizer.eos_token_id # Stop at eos
)
```

Back to our storyteller:
- `repetition_penalty=1.2`: The storyteller avoids using the same words or phrases too often, making the story more engaging.
- `eos_token_id=tokenizer.eos_token_id`: The storyteller knows when to stop, so the story doesn't drag on unnecessarily.

This combination keeps the story interesting, concise and well-paced!

```txt
The future of Articial Intelligence is in doubt, 
and the only way to stop it from becoming a reality 
will be through an unprecedented level-headed approach 
that includes both scientific research (as well as public relations) 
on issues such for example education policy.
I hope this article has been useful at all levels - especially 
when you're trying so hard to keep up with new technology 
like Facebook's AI platform!
``` 