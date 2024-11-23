import goodfire
import openai
from typing import List
import re
from prompts import (
    RETRIEVER_SYSTEM_PROMPT,
)
from goodfire import FeatureGroup
import os
from custom_decorators import deprecated


class Retriever:
    
    def __init__(self):
        raise NotImplementedError("Use from_goodfire() or from_separate_clients() instead")
    
    @classmethod #TODO debug
    def from_goodfire(cls, client: goodfire.Client, variant: str):
        """Create a Retriever from a GoodFire client.

        Args:
            client (goodfire.Client): the GoodFire client
            variant (str): model name to use for both the prompter and feature searcher
        """
        assert isinstance(client, goodfire.Client), "client must be a goodfire.Client"
        assert isinstance(variant, str), "variant must be a string"
                
        self = cls.__new__(cls)
        self.system_prompt = RETRIEVER_SYSTEM_PROMPT
        self.prompter = client
        self.prompter_variant = goodfire.Variant(variant)
        self.feature_searcher = client
        self.feature_searcher_variant = goodfire.Variant(variant)
        return self
    
    @classmethod #TODO debug
    def from_separate_clients(cls, prompter: openai.Client, prompter_variant: str, feature_searcher: goodfire.Client, feature_searcher_variant: str):
        """Create a Retriever from separate clients for the prompter and feature searcher.

        Args:
            prompter (openai.Client): the prompter client from OpenAI
            prompter_variant (str): model name to use for the prompter
            feature_searcher (goodfire.Client): the feature searcher client from GoodFire
            feature_searcher_variant (str): model name to use for the feature searcher
        """
        assert isinstance(prompter, openai.Client), "prompter must be an openai.Client"
        assert isinstance(prompter_variant, str), "prompter_variant must be a string"
        assert isinstance(feature_searcher, goodfire.Client), "feature_searcher must be a goodfire.Client"
        assert isinstance(feature_searcher_variant, str), "feature_searcher_variant must be a string"
        
        self = cls.__new__(cls)
        self.system_prompt = RETRIEVER_SYSTEM_PROMPT
        self.prompter = prompter
        self.prompter_variant = prompter_variant
        self.feature_searcher = feature_searcher
        self.feature_searcher_variant = goodfire.Variant(feature_searcher_variant)
        return self
    
    def retrieve_features(self, target_behavior: str, k: int = 10) -> FeatureGroup:
        """Retrieve features relevant to a given prompt.

        Args:
            target_behavior (str): The desired behavior to evaluate against
        Returns:
            FeatureGroup: A feature group object containing the feature values retrieved from the search
        """
        features, relevance = self.feature_searcher.features.search(
                target_behavior, model=self.feature_searcher_variant, top_k=k
            )
        return features
    
    @deprecated
    def retrieve_features_deprecated(self, target_behavior: str, critic: str = None, k: int = 10):
        """Retrieve features relevant to a given prompt.

        Args:
            target_behavior (str): The desired behavior to evaluate against
        Returns:
            List[features]: List of feature values retrieved from the search
        """

        content = f"Target Behavior:\n{target_behavior}\n\n"
        if critic:
            content += f"Critique:\n{critic}\n\n"

        completion = self.prompter.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": RETRIEVER_SYSTEM_PROMPT,
                },
                {
                    "role": "system",
                    "content": content,
                },
            ],
            model=self.prompter_variant,
            # stream=True,
            # max_completion_tokens=200,
        )
        if isinstance(self.prompter, goodfire.Client):
            completion = completion.choices[0].message["content"]
        else:
            completion = completion.choices[0].message.content

        queries = re.findall(r"Q: (.*)", completion)
        for query in queries:
            print(f"{query}")

        all_features = []

        for query in queries:
            features, relevance = self.feature_searcher.features.search(
                query, model=self.feature_searcher_variant, top_k=k
            )
            all_features.extend(features)
        return list(set(all_features))
