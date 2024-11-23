import goodfire
from typing import List
import re
from prompts import (
    SCORER_SYSTEM_PROMPT,
)
from goodfire import FeatureGroup
import os
from custom_decorators import deprecated

#TODO: implement openAI scorer
class Scorer:
    def __init__(self, client: goodfire.Client, variant: str, scale: float = 1.0):
        
        self.client = client
        self.variant = goodfire.Variant(variant)
        self.accumulated_prompts = []
        self.log_file = "scorer_logs.txt"
        self.scale = scale
        assert scale >= 1.0

    def parseStrToList(self, score_gen: str):
        numbers = re.findall(r"-?\d*\.?\d+", score_gen)
        weights = [float(x) for x in numbers]  # Convert strings to floats
        return weights
    
    def score_features(
        self, target_behavior: str, critique: str, features: FeatureGroup, prev_scores: List[float] = []
    ):
        """Score a list of features to get their weights.

        Args:
            target_behavior (str): The desired behavior to evaluate against
            critique (str): A free-text critique evaluating how well the steered
                output matches the target behavior given the input
            features (List[float]): List of feature values to score

        Returns:
            List[float]: Ordered list of feature weights between -1 and 1
        """
        #TODO: implement scale (change system prompt)
        #scale = 5 * self.scale
        scale = 1.0

        #building the prompt
        #start with the system prompt 
        self.SYS_PROMPT = SCORER_SYSTEM_PROMPT.format(target_behavior=target_behavior)
        
        if not (critique is None or len(critique) == 0):
            critique = (
                f"""The feedback is {critique}, adjust the score to improve critique."""
            )
        else:
            critique = ""

        if len(prev_scores) > 0 and len(critique) > 0:
            self.accumulated_prompts.append(
                {"role": "assistant", "content": str(prev_scores)}
            )
            self.accumulated_prompts.append({"role": "user", "content": critique})

        prompt = [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": f"Features:\n{str(features)}\n\n"},
        ]
        prompt += self.accumulated_prompts
        
        score_gen = ""
        for token in self.client.chat.completions.create(
            prompt,
            model=self.variant,
            stream=True,
            max_completion_tokens=50,
        ):
            score_gen += token.choices[0].delta.content

        print(f"=== Scorer: New Scoring Session ===")
        print(f"{prompt=}")
        print(f"{score_gen=}")
        weights = self.parseStrToList(score_gen)
        weights = [float(x) / scale for x in weights]

        while len(weights) != len(features):
            print(f"Length of scores does not match length of features. {weights} {len(features)=}")
            prompt += [
                    {"role": "assistant", "content": score_gen},
                    {
                        "role": "user",
                        "content": "Please answer only with a python list of scores where the length of scores is the length of features.",
                    },
                ]
            score_gen = ""
            for token in self.client.chat.completions.create(
                messages=prompt, 
                model=self.variant,
                stream=True,
                max_completion_tokens=50,
            ):
                score_gen += token.choices[0].delta.content
            print(f"=== Scorer: New Scoring Session ===")
            print(f"{prompt=}")
            print(f"{score_gen=}")
                
            weights = self.parseStrToList(score_gen)
            weights = [float(x) / scale for x in weights]
        print(f"final weights for Scorer: {weights}")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Scorer: New Scoring Session ===\n")
            f.write("Prompt:\n")
            for p in prompt:
                f.write(f"{p}\n")
            f.write(f"Response:\n{score_gen}\n")
            f.write("=" * 50)

        return weights

    @deprecated
    def score_features_deprecated(
        self, target_behavior: str, critique: str, features: FeatureGroup, prev_scores: List[float] = []
    ):
        """Score a list of features to get their weights.

        Args:
            target_behavior (str): The desired behavior to evaluate against
            critique (str): A free-text critique evaluating how well the steered
                output matches the target behavior given the input
            features (List[float]): List of feature values to score

        Returns:
            List[float]: Ordered list of feature weights between -1 and 1
        """

        scale = 5 * self.scale

        if not (critique is None or len(critique) == 0):
            critique = (
                f"""The feedback is {critique}, adjust the score to improve critique."""
            )
        else:
            critique = ""

        if len(prev_scores) > 0 and len(critique) > 0:
            self.accumulated_prompts.append(
                {"role": "assistant", "content": str(prev_scores)}
            )
            self.accumulated_prompts.append({"role": "user", "content": critique})

        # self.SYS_PROMPT = f"""Let's play a game called the backpropagation game- This game is very important. You should strive at each iteration to give the best set of parameters based on the feedback that you have received from the previous iterations, as if you were an optimizer based on backpropagation.\nYou are given a list of features, and explanations of what they mean. Your aim is to choose the right combination of feature values to reach this desired model behavior: {target_behavior}. Give it all your best.\n\nRemember that the meaning of each feature may be informative in telling you how you should steer these features, but you should strongly consider the feedback that you have received in previous rounds for steering features in a certain way. For instance, you may have steered a feature too much and then the output of the model may become nonsensical, or not right for the input prompt. In each round, output the value that you want to assign to each feature using a list of scores between -1 and 1. Give it as a python List of features. You should return a value for each feature.\nExample: for 5 features, you should output a python list of 5 features, such as [0.3, -0.7, 0.1, 0.9, 0.9]."""

        self.SYS_PROMPT = SCORER_SYSTEM_PROMPT

        content = (
            f"\nFeatures:\n{str(features)}\nTarget Behavior:\n{target_behavior}\n\n"
        )
        if prev_scores:
            prev_scores = [int(x * scale) for x in prev_scores]
            content += f"Previous Scores:\n{str(prev_scores)}\n"
        if critique:
            content += f"Critique:\n{critique}\n\n"

        prompt = [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": content},
        ]

        # prompt = base_prompts  # + self.accumulated_prompts
        completion = self.client.chat.completions.create(
            messages=prompt,
            model=self.variant,
            # stream=True,
            # max_completion_tokens=200,
        )
        if isinstance(self.client, goodfire.Client):
            score_gen = completion.choices[0].message["content"]
        else:
            score_gen = completion.choices[0].message.content

        print(f"{score_gen=}")
        weights = self.parseStrToList(score_gen)
        weights = [float(x) / scale for x in weights]

        while len(weights) != len(features):
            print(f"Length of scores does not match length of features. {weights}")
            score_gen = self.client.chat.completions.create(
                messages=prompt
                + [
                    {"role": "assistant", "content": score_gen},
                    {
                        "role": "user",
                        "content": "Please answer only with a python list of scores where the length of scores is the length of features.",
                    },
                ],
                model=self.variant,
                stream=False,
                # max_completion_tokens=250,
            )
            if isinstance(self.client, goodfire.Client):
                score_gen = completion.choices[0].message["content"]
            else:
                score_gen = completion.choices[0].message.content
            weights = self.parseStrToList(score_gen)
            weights = [float(x) / scale for x in weights]
        print(f"{weights}")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Scorer: New Scoring Session ===\n")
            f.write("Prompt:\n")
            for p in prompt:
                f.write(f"{p}\n")
            f.write(f"Response:\n{score_gen}\n")
            f.write("=" * 50)

        return weights