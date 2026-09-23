# Health Context Challenge

Thank you for doing this small homework challenge 🙇

The work you do here will also be presented during the onsite portion of the interview, so while AI tooling is allowed it is _critical_ to deeply understand the material you will be sharing. You will be presenting to both technical and non-technical teammates and need to be able to explain the _why_ and _how_.

I hope this ends up being fun!

## The Task

1. **Fork this repo** and use it as the basis for the project.
2. **Compact context.** Build a runnable script, notebook, or pipeline that turns a patient's HIE pulls into a compact, useful context of at most 900,000 tokens, counted as described in [Token counting](#token-counting).
   - consider what can be cut entirely vs. what needs to be summarized
   - consider what can be deterministic vs LLM summaries
3. **Evals.** Build an eval/test suite that measures how good that context is. You decide what "good" means, and you defend the choice.
4. **Retrieval.** Build a RAG/lookup system for whatever doesn't fit in the compact context.

### Constraints

- It has to work and be demoable. It doesn't have to be pretty.
- Use any AI tooling you like, but you must fully own the result. Expect us to ask about any line of the code and any part of the output.
- We can give you Claude API keys. You can use any provider you prefer.
- Use your favorite and most familiar libraries to get work done (we happen to use `dagster` for etl and `pydantic-ai` for interfacing with llms, if you have no preference, but seriously use what makes you comfortable!)

### Example eval questions

These are here for inspiration only, you can phrase your evals in any way you like, just be prepared to explain it. They aren't a spec, and we don't expect you to use them. All three are about `Merlene950_Marlin805_Thompson596_*.json`.

- **What was her most recent HbA1c, and when was it taken?** Checks that one exact value and its date survive, out of eight readings taken over nine years.
- **Is her blood sugar getting worse? Tie it to any related diagnosis.** Checks a trend over time and whether the context links a lab result to a diagnosis.
- **What is she allergic to?** She has no allergies on record, so the correct answer is "nothing recorded." Checks whether a model with a thin context invents one.

### The Data

The synthetic HIE data is described in the [DATA.md](DATA.md). The largest bundle is \~52 MB, so you see why we need the compression!

### Token counting

A context is the exact text you would send to the model, and you count it as one user message with Anthropic's token counting endpoint on `claude-opus-5`:

```python
from anthropic import Anthropic

tokens = Anthropic().messages.count_tokens(
    model="claude-opus-5",
    messages=[{"role": "user", "content": context}],
).input_tokens
assert tokens <= 900_000
```

The cap is below the model's 1M-token window so there's room left for a question and its answer. This is the budget even if you use a different provider. Other tokenizers, `tiktoken` included, give different counts for the same text.

## What you present

Walk us through the design, the architecture, and the implementation. Tell us which choices were _good_, and which were _compromises_, and—of course—why.

During the onsite we will also extend some of the ideas from [The Task](#the-task). e.g. can we get the context to be super small? can we generalize retrevials for arbitrary labs? the sky is the limit 🪁

## Other stuff

The golden rule is to spend your time doing what you know and with tools you are familiar with.

- `mise.toml` is an environment management tool we use similar to `asdf` and `dotenv`. Completely optional! Ignore it if you want.
- `hk.pkl` is a linter runner, similar to `pre-commit`. Completely optional! Ignore it if you want.
- `pyproject.toml` preferred way to manage dependencies, but you can use whatev
