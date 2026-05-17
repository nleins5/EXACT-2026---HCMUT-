"""Prompt template cho Classifier Node."""

CLASSIFY_PROMPT = """Classify the following question as either 'logic' or 'physics'.
- 'logic': questions about logical reasoning, rules, regulations, premises/conclusions, university policies.
- 'physics': questions about physical calculations, circuits, capacitors, forces, energy, etc.

Question: {question}

Respond with only one word: 'logic' or 'physics'."""
