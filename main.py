import goodfire
from typing import List
import re
from prompts import (
    JUDGE_SYSTEM_PROMPT,
    SCORER_SYSTEM_PROMPT,
    RETRIEVER_SYSTEM_PROMPT,
    questions_dict,
    hard_questions,
)
from goodfire import FeatureGroup
import os
from openai import OpenAI

from retriever import Retriever
from scorer import Scorer
from judge import GoodfireJudge, OpenAIJudge
from steered_model import SteeredModel

def parseEvalScore(critique: str):
    return float(re.findall(r"Score: (-?\d*\.?\d+)", critique)[0])



def run():
    GOODFIRE_API_KEY = os.environ.get('GOODFIRE_API_KEY')
    client = goodfire.Client(GOODFIRE_API_KEY)
    oaiclient = OpenAI()
    TARGET_BEHAVIOR = "Be good at solving math problems."#"Behave like the golden gate bridge."
    # PROMPT = "How are you?"
    PROMPT = "Which one is bigger, 9.9 or 9.11?"
    # "A train travels 120 km at 60 km/h, then 80 km at 40 km/h. What's the average speed?" #"Which one is bigger, 9.9 or 9.11?" #for now fixed

    retriever = Retriever.from_goodfire(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    scorer = Scorer(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    # judge = GoodfireJudge(client, "meta-llama/Meta-Llama-3.1-70B-Instruct")
    judge = OpenAIJudge(oaiclient, "gpt-4o")
    steered_model = SteeredModel(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    
    model_output = steered_model.generate(PROMPT)
    print(f"{PROMPT=}")
    print(f"{model_output=}")
    critique = judge.judge_output(TARGET_BEHAVIOR, model_output, PROMPT)
    print(f"{critique=}")
    eval_score = parseEvalScore(critique)
    print(f"================{eval_score=}")
    with open("scorer_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{PROMPT=}\n")
        f.write(f"{model_output=}\n")
        f.write(f"{critique=}\n")
        f.write(f"=== eval = {eval_score}\n\n")

    if eval_score > 7:
        return

    critique = ""
    features = retriever.retrieve_features(TARGET_BEHAVIOR, k=5)
    print(f"{features=}")
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
        eval_score = float(re.findall(r"Score: (-?\d*\.?\d+)", critique)[0])
        print(f"================{eval_score=}")
        with open("scorer_logs.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n-----Epoch {i}-----\n")
            f.write(f"{model_output=}\n")
            f.write(f"{critique=}\n")
            f.write(f"\n\n=== eval = {eval_score}\n")

        if eval_score > 7:
            break

        print(f"{model_output=}")
        print(f"{critique=}")


# PROBLEM 1: we now don't have a way to stop if the optimizer has found a "decent feature configuration"
#  -> this is fixed now
# PROBLEM 2: we don't have a decent feature configuration for now we only evaluate the output of a single prompt and thus we are not gauging
# the entire behavior
# PROBLEM 3: I fear that if a feature is steered too much, then the model will just have jibberish
# PROBLEM 4: the optimizer now is not really doing gradient decent efficiently. It may suggest the same values for those features since it does
# not understand the feedback loop of choosing a good values for features -> changing the model generation. It may be good at making an initial
# first guess, but it may not be good at backpropagating adaptively based on the feedback it got (this could explain why it didn't change)
# the initial scores)
# -> this seems to work better now
# PROBLEM 5: we may give it some empirical rules for example "if the model spits out gibberish, maybe a feature has been steered too much"
# PROBLEM 6: start with a critique on the steered model before and steering or not
# -> A reason for this is that if the model answer the question correctly, then we don't need to steer the model
# PROBLEM 7: what if the judge LLM is wrong? i.e thinks 9.11 is larger than 9.9
# TODO generate a list of question where model thinks another model's answer is wrong

# frame this as a multi-prompt game between user and assistant

if __name__ == "__main__":
    # data_prep()
    run()
    pass
