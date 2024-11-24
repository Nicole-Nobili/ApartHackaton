import goodfire
from typing import List
import re
import openai
from prompts import (
    JUDGE_SYSTEM_PROMPT,
)
from custom_decorators import deprecated
from abc import ABC, abstractmethod

class Judge(ABC):
    def __init__(self, variant: str):
        self.SYS_PROMPT = JUDGE_SYSTEM_PROMPT
        self.variant = variant
        self.log_file = "class_logs.txt"
        
    @abstractmethod
    def judge_output(self, target_behavior: str, steered_model_output: str, steered_model_input: str) -> str:
        pass

class GoodfireJudge(Judge):
    def __init__(self, client: goodfire.Client, variant: str):
        super().__init__(variant=variant)
        assert isinstance(client, goodfire.Client), "client must be a goodfire.Client"
        self.client = client
        self.variant = goodfire.Variant(variant)

    def judge_output(self, target_behavior: str, steered_model_output: str, steered_model_input: str, max_tokens: int = 1024) -> str:
        completion = ""
        for token in self.client.chat.completions.create(
            [
                {"role": "system", "content": self.SYS_PROMPT},
                {"role": "user", "content": f"""Input prompt:\n{steered_model_input}\n\nResponse:\n{steered_model_output}\n\nTarget Behavior:{target_behavior}\n\n"""}
            ],
            model=self.variant,
            stream=True,
            max_completion_tokens=max_tokens,
        ):
            completion += token.choices[0].delta.content
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Judge: New Judging Session ===\n")
            f.write("Steered Model Input:\n")
            f.write(f"{steered_model_input}\n")
            f.write("Steered Model Output:\n")
            f.write(f"{steered_model_output}\n")
            f.write("Critique:\n")
            f.write(f"{completion}\n")
            f.write("=" * 50)
        return completion

class OpenAIJudge(Judge):
    def __init__(self, client: openai.Client, variant: str):
        super().__init__(variant=variant)
        assert isinstance(client, openai.Client), "client must be an openai.Client"
        self.client = client

    def judge_output(self, target_behavior: str, steered_model_output: str, steered_model_input: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.variant,
            messages=[
                {"role": "system", "content": self.SYS_PROMPT},
                {"role": "user", "content": f"""Input prompt:\n{steered_model_input}\n\nResponse:\n{steered_model_output}\n\nTarget Behavior:{target_behavior}\n\n"""}
            ],
            max_tokens=max_tokens
        )
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Judge: New Judging Session ===\n")
            f.write("Steered Model Input:\n")
            f.write(f"{steered_model_input}\n")
            f.write("Steered Model Output:\n")
            f.write(f"{steered_model_output}\n")
            f.write("Critique:\n")
            f.write(f"{response.choices[0].message.content}\n")
            f.write("=" * 50)
        return response.choices[0].message.content