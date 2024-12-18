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
        self.variant = variant

    @abstractmethod
    def judge_output(self, target_behavior: str, steered_model_output: str, steered_model_input: str) -> str:
        pass

class GoodfireJudge(Judge):
    def __init__(self, client: goodfire.Client, variant: str, sys_prompt: str = JUDGE_SYSTEM_PROMPT):
        super().__init__(variant=variant)
        assert isinstance(client, goodfire.Client), "client must be a goodfire.Client"
        self.client = client
        self.variant = goodfire.Variant(variant)
        self.SYS_PROMPT = sys_prompt

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
        return completion

class OpenAIJudge(Judge):
    def __init__(self, client: openai.Client, variant: str, sys_prompt: str = JUDGE_SYSTEM_PROMPT):
        super().__init__(variant=variant)
        assert isinstance(client, openai.Client), "client must be an openai.Client"
        self.client = client
        self.SYS_PROMPT = sys_prompt

    def judge_output(self, target_behavior: str, steered_model_output: str, steered_model_input: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.variant,
            messages=[
                {"role": "system", "content": self.SYS_PROMPT},
                {"role": "user", "content": f"""Input prompt:\n{steered_model_input}\n\nResponse:\n{steered_model_output}\n\nTarget Behavior:{target_behavior}\n\n"""}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content