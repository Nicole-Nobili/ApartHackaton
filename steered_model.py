import goodfire
from typing import List
from custom_decorators import deprecated

class SteeredModel:
    def __init__(self, client: goodfire.Client, variant: str):
        self.client = client
        self.variant = goodfire.Variant(variant)

    def set_features(self, features, scores: List[float]):
        """Set the features and scores for the steered model.

        Args:
            features (FeatureGroup): List of feature values
            scores (List[float]): List of scores between -1 and 1 for the corresponding features
        """
        print(f"{features=}, {scores=}")

        assert len(features) == len(scores)
        self.variant.reset()
        for feature, score in zip(features, scores):
            self.variant.set(feature, score)

    def generate(self, prompt: str):
        """Generate a response from the steered model.

        Args:
            prompt (str): The input prompt to the model
        """
        completion = ""
        for token in self.client.chat.completions.create(
            [{"role": "user", "content": prompt}],
            model=self.variant,
            stream=True,
            max_completion_tokens=200,
        ):
            completion += token.choices[0].delta.content
        return completion