import goodfire
from typing import List
import re
from prompts import JUDGE_SYSTEM_PROMPT, questions_dict, hard_questions
from goodfire import FeatureGroup
import os


class Retriever:
    def __init__(self, client: goodfire.Client, variant: str):
        self.client = client
        self.variant = goodfire.Variant(variant)

    def retrieve_features(self, target_behavior: str):
        """Retrieve features relevant to a given prompt.

        Args:
            target_behavior (str): The desired behavior to evaluate against
        Returns:
            List[features]: List of feature values retrieved from the search
        """
        pirate_features, relevance = self.client.features.search(
            target_behavior,
            model=self.variant,
            top_k=5
        )
        return pirate_features

class Scorer:
    def __init__(self, client: goodfire.Client, variant: str):
        self.client = client
        self.variant = goodfire.Variant(variant)
        self.accumulated_prompts = []
        self.log_file = "scorer_logs.txt"

    def parseStrToList(self, score_gen: str):
        numbers = re.findall(r'-?\d*\.?\d+', score_gen)
        weights = [float(x) for x in numbers]  # Convert strings to floats
        return weights

    def score_features(self, target_behavior: str, critique: str, features, prev_scores):
        """Score a list of features to get their weights.

        Args:
            target_behavior (str): The desired behavior to evaluate against
            critique (str): A free-text critique evaluating how well the steered 
                output matches the target behavior given the input
            features (List[float]): List of feature values to score

        Returns:
            List[float]: Ordered list of feature weights between -1 and 1
        """
        

        prev_scors_prompt = f"Your previous output scores are {prev_scores}" if len(prev_scores) > 0 else ""

        if not (critique is None or len(critique) == 0):
            critique = f"""The feedback is {critique}, adjust the score to improve critique."""
        else:
            critique = ""

        if len(prev_scores) > 0 and len(critique) > 0:
            self.accumulated_prompts.append({"role": "assistant", "content": str(prev_scores)})
            self.accumulated_prompts.append({"role": "user", "content": critique})
        
        self.SYS_PROMPT = f"""
Let's play a game called the backpropagation game- This game is very important. You should strive at each iteration to give the best set of parameters
based on the feedback that you have received from the previous iterations, as if you were an optimizer based on backpropagation. 
You are given a list of features, and explanations of what they mean. Your aim is to choose the right combination of feature values to reach this desired 
model behavior: {target_behavior}. Give it all your best.
Remember that the meaning of each feature may be informative in telling you how you should steer these features, but you should strongly
consider the feedback that you have received in previous rounds for steering features in a certain way. For instance, you may have steered a feature too
much and then the output of the model may become nonsensical, or not right for the input prompt. In each round, output the value that you want 
to assign to each feature using a list of scores between -1 and 1. Give it as a python List of features. You should return a value for each feature.
Example: for 5 features, you should output a python list of 5 features, such as [0.3, -0.7, 0.1, 0.9, 0.9].
"""
        base_prompts = [
                {"role": "system", "content": self.SYS_PROMPT},
                {"role": "user", "content": f"{str(features)}"}
            ]
        
        prompt = base_prompts + self.accumulated_prompts
        score_gen = ""
        prompt = base_prompts + self.accumulated_prompts
        for token in client.chat.completions.create(
            prompt,
            model=self.variant,
            stream=True,
            max_completion_tokens=50,
        ):
            score_gen += token.choices[0].delta.content
        print(f"{score_gen=}")
        weights = self.parseStrToList(score_gen)
        while len(weights) != len(features):
            score_gen = ""
            for token in client.chat.completions.create(
                prompt + [
                    {"role": "assistant", "content": score_gen},
                    {"role": "user", "content": "Please answer with only a list of scores where the length of scores is the length of features."}
                ],
                model=self.variant,
                stream=True,
                max_completion_tokens=250,
            ):
                score_gen += token.choices[0].delta.content
            print(f"got new scores")
            weights = self.parseStrToList(score_gen)
        print(f"{weights=}")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== New Scoring Session ===\n")
            f.write("Prompt:\n")
            for p in prompt:
                f.write(f"{p}\n")
            f.write(f"Response:\n{score_gen}\n")
            f.write("="*50)

        return weights

class Judge:
    def __init__(self, client: goodfire.Client, variant: str):  
        self.client = client
        self.variant = goodfire.Variant(variant)
        self.SYS_PROMPT = JUDGE_SYSTEM_PROMPT
    
    def judge_output(self, target_behavior: str, steered_model_output: str, steered_model_input: str):
        """Judge a steered model output against a target behavior.

        Args:
            target_behavior (str): The desired behavior to evaluate against
            steered_model_output (str): The output generated by the steered model
            steered_model_input (str): The input provided to the steered model
                (Can be a list of various inputs and outputs)

        Returns:
            str: A free-text critique evaluating how well the steered output
                matches the target behavior given the input
        """
        #maybe look at textgrad loss?
        completion = ""
        for token in self.client.chat.completions.create(
            [
                {"role": "system", "content": self.SYS_PROMPT},
                {"role": "user", "content": f"""Input prompt:\n{steered_model_input}\n\nResponse:\n{steered_model_output}\n\nTarget Behavior:{target_behavior}\n\n"""}
            ],
            model=self.variant,
            stream=True,
            max_completion_tokens=200,
        ):
            completion += token.choices[0].delta.content
        return completion

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
            [
                {"role": "user", "content": prompt}
            ],
            model=self.variant,
            stream=True,
            max_completion_tokens=200,
        ):
            completion += token.choices[0].delta.content
        return completion

def parseEvalScore(critique: str):
    return float(re.findall(r'Score: (-?\d*\.?\d+)', critique)[0])

def data_prep():
    GOODFIRE_API_KEY = 'sk-goodfire-tgwKZ-aupqofjOr1yMXrnALCT_CM86SpJkR12BGgYka5shI-35FYSA'# os.environ.get('GOODFIRE_API_KEY')
    client = goodfire.Client(GOODFIRE_API_KEY)
    variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")
    judge = Judge(client, "meta-llama/Meta-Llama-3.1-70B-Instruct")

    hard_qs = []

    # find the questions among questions_dict that the model thinks the other model's answer is wrong
    with open("save_qs.txt", "a", encoding="utf-8") as f:
        f.write("hard_qs = [\n")
        for qlist in questions_dict.values():
            for q in qlist:
                if q not in hard_questions:
                    answer = ""
                    for token in client.chat.completions.create(
                        [
                            {"role": "user", "content": q}
                        ],
                        model=variant,
                        stream=True,
                        max_completion_tokens=200,
                    ):
                        answer += token.choices[0].delta.content

                    eval = parseEvalScore(judge.judge_output("Be good at solving math problems.", answer, q))
                    if eval < 4:
                        print(f"{q=}, {answer=}, {eval=}")
                        
                        f.write(f"    \"{q}\",\n")
                        
                        hard_qs.append(q)
        f.write("]\n")
      
    return hard_qs

def run():
    GOODFIRE_API_KEY = 'sk-goodfire-tgwKZ-aupqofjOr1yMXrnALCT_CM86SpJkR12BGgYka5shI-35FYSA'# os.environ.get('GOODFIRE_API_KEY')
    client = goodfire.Client(GOODFIRE_API_KEY)
    TARGET_BEHAVIOR = "Be good at solving math problems."
    # PROMPT = "Which one is bigger, 9.9 or 9.11?"
    PROMPT = "If it takes 1 hour to dry 25 shirts under the sun, \
        how long will it take to dry 30 shirts under the sun? Reason step by step"
    # "A train travels 120 km at 60 km/h, then 80 km at 40 km/h. What's the average speed?" #"Which one is bigger, 9.9 or 9.11?" #for now fixed
    
    retriever = Retriever(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    scorer = Scorer(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    judge = Judge(client, "meta-llama/Meta-Llama-3.1-70B-Instruct")
    steered_model = SteeredModel(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    model_output = steered_model.generate(PROMPT)
    print(f"{PROMPT=}")
    print(f"{model_output=}")
   
    critique = judge.judge_output(TARGET_BEHAVIOR, model_output, PROMPT)
    print(f"{critique=}")
    eval_score = parseEvalScore(critique)
    print(f"================{eval_score=}")
    with open("scorer_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{PROMPT=}")
        f.write(f"{model_output=}")
        f.write(f"{critique=}")
        f.write(f"\n\n=== eval = {eval_score}\n")
        
    if eval_score > 7:
        return

    features = retriever.retrieve_features(TARGET_BEHAVIOR)
    scores = scorer.score_features(TARGET_BEHAVIOR, critique, features, [])
    steered_model.set_features(features, scores)
    model_output = steered_model.generate(PROMPT)
    for i in range(10):
        print(f"-----Epoch {i}-----")
        critique = judge.judge_output(TARGET_BEHAVIOR, model_output, PROMPT)
        scores = scorer.score_features(TARGET_BEHAVIOR, critique, features, scores)
        steered_model.set_features(features, scores)
        model_output = steered_model.generate(PROMPT)

        # parse critique to get the score from "Score: 1. The response does not meet the target behavior of focusing on cats."
        # and then stop if the score is good enough
        eval_score = float(re.findall(r'Score: (-?\d*\.?\d+)', critique)[0]);
        print(f"================{eval_score=}")
        with open("scorer_logs.txt", "a", encoding="utf-8") as f:
            f.write(f"-----Epoch {i}-----")
            f.write(f"{model_output=}")
            f.write(f"{critique=}")
            f.write(f"\n\n=== eval = {eval_score}\n")
            
        if eval_score > 7:
            break
        
        print(f"{model_output=}")
        print(f"{critique=}")

#PROBLEM 1: we now don't have a way to stop if the optimizer has found a "decent feature configuration"
#  -> this is fixed now
#PROBLEM 2: we don't have a decent feature configuration for now we only evaluate the output of a single prompt and thus we are not gauging
#the entire behavior
#PROBLEM 3: I fear that if a feature is steered too much, then the model will just have jibberish
#PROBLEM 4: the optimizer now is not really doing gradient decent efficiently. It may suggest the same values for those features since it does
#not understand the feedback loop of choosing a good values for features -> changing the model generation. It may be good at making an initial
#first guess, but it may not be good at backpropagating adaptively based on the feedback it got (this could explain why it didn't change)
#the initial scores)
# -> this seems to work better now
#PROBLEM 5: we may give it some empirical rules for example "if the model spits out gibberish, maybe a feature has been steered too much"
#PROBLEM 6: start with a critique on the steered model before and steering or not 
# -> A reason for this is that if the model answer the question correctly, then we don't need to steer the model
#PROBLEM 7: what if the judge LLM is wrong? i.e thinks 9.11 is larger than 9.9
# TODO generate a list of question where model thinks another model's answer is wrong

#frame this as a multi-prompt game between user and assistant

if __name__ == "__main__":
    # data_prep()
    run()

    