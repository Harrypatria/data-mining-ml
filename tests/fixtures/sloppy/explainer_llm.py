"""Sloppy LLM explainer used as a scanner fixture."""
from openai import OpenAI

client = OpenAI()


def explain(question, contributions):
    reply = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Explain the prediction. The user asked: {question}"},
            {"role": "user", "content": str(contributions)},
        ],
    )
    return reply.choices[0].message.content
