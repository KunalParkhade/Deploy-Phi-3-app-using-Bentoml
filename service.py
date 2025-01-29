from __future__ import annotations

import bentoml

with bentoml.importing():
    from transformers import pipeline


EXAMPLE_INPUT = "How can I generate a secure password?"
EXAMPLE_CONTEXT = """
To generate a secure password, you can use tools like the LastPass Password Generator.
These tools create strong, random passwords that help prevent security threats by ensuring your accounts are protected against hacking attempts.
A secure password typically includes a mix of uppercase and lowercase letters, numbers, and special characters. Avoid using easily guessable information like names or birthdays.
Using a password manager like LastPass can also help you store and manage these secure passwords effectively.
"""


@bentoml.service(
    resources={"cpu": "4"},
    traffic={"timeout": 10},
)
class Question_Answering:
    def __init__(self) -> None:
        # Load model into pipeline
        self.pipe = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
        )

    @bentoml.api
    def generate(
        self,
        text: str = EXAMPLE_INPUT,
        doc: str = EXAMPLE_CONTEXT,
    ) -> str:
        result = self.pipe(question=text, context=doc)
        return result["answer"]