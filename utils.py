import goodfire
from typing import Str, List

class Retriever:
    def __init__(self, client: goodfire.Client, variant: str):
        self.client = client
        self.variant = variant

    def retrieve_features(self, prompt: Str):
        """Retrieve features relevant to a given prompt.

        Args:
            prompt (str): The input prompt to search features for

        Returns:
            List[float]: List of feature values retrieved from the search
        """
        pirate_features, relevance = self.client.features.search(
            prompt,
            model=self.variant,
            top_k=5
        )
        return pirate_features

class Scorer:
    def __init__(self, client: goodfire.Client, variant: str):
        self.client = client
        self.variant = variant
    
    def score_features(self, prompt: Str, features: List[float], ):
        """Score a list of features to get their weights.

        Args:
            prompt (str): The input prompt which is original to the user score against
            features (List[float]): List of feature values to score

        Returns:
            List[float]: Ordered list of feature weights between -1 and 1
        """
        weights = []
        return weights
    
class Judge:
    
    #input is the output of the steer LLM, output is a critique of the output respect to the input"
    #so your inputs are threefold: 
    #1. the target behavior
    #2. steered model output
    #3. steered model input (these can always be a list of various inputs and outputs)
    
    #and your outputs are: a free text (no structure) critique of the output respect to the target behavior
    