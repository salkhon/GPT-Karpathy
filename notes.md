# Observations

> chatgpt generates output left to right
> output varies from run to run
> for any prompt it gives different outputs

It is what we call a **language model**. It models the sequence of words, characters or tokens and it knows how words follow eachother in the English language. From it's perspective -

> It's completing the sequence

The prompt gives it the start of the sequence, and it completes the outcome. So it's a language model in the sense.

> Here we will focus on under the hood components of what makes chatgpt work.

> What is the neural net under the hood that models the sequence of these words?

### GPT: Generatively Pre-trained Transformer

> We would like to produce somethign like chatgpt.

Ofcourse nothing quite of its quality - because it's trained on a great chunk of the WWW. There is a lot of pretraining, finetuning for that.

> We want to train a transformer based language model.

> It our case it will be a character level language model

We work with _tiny shakespear_. All of shakespear.

We model how these texts follow eachother.

> The transformer will look at the text and predict that 'g' (or any other) character is likely to come next.

Goal output: Transformer will learn the essence of shakespear, and generate character by character. ChatGPT has tokens of "subwords". 