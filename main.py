import goodfire
from typing import List
import re
from prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT1,
    SCORER_SYSTEM_PROMPT,
    RETRIEVER_SYSTEM_PROMPT,
    questions_dict,
    hard_questions,
)
from goodfire import FeatureGroup
import os
from openai import OpenAI

from retriever import Retriever
from scorer import Scorer, ScorerWithoutCritique
from judge import GoodfireJudge, OpenAIJudge
from steered_model import SteeredModel
from datetime import datetime
from logger import ConcurrentLogHandler, logger, logging
import concurrent.futures

def parseEvalScore(critique: str):
    found = re.findall(r"Score: (-?\d*\.?\d+)", critique)
    if len(found) > 0:
        eval_score = float(found[0])
    else:
        eval_score = 0 ## error, no score found
    return eval_score

# Configure shared epoch evaluation logger
epoch_eval_logger = logging.getLogger('epoch_eval')
epoch_eval_logger.propagate = False  # Prevent propagation to root logger
epoch_eval_logger.setLevel(logging.INFO)
epoch_eval_handler = ConcurrentLogHandler('epoch_eval_logs.txt')
epoch_eval_logger.addHandler(epoch_eval_handler)

def run(prompt, judge_sys_prompt=JUDGE_SYSTEM_PROMPT, log_prefix="", num_features=5, epoch=10):
    GOODFIRE_API_KEY = os.environ.get('GOODFIRE_API_KEY')
    client = goodfire.Client(GOODFIRE_API_KEY)
    oaiclient = OpenAI()
    # TARGET_BEHAVIOR = "Correctly answer the questions in the prompt."

    # response = oaiclient.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": "Please respond with instructions for how another model could best answer user's question in less than 100 characters."},
    #         {"role": "user", "content": f"""{prompt}"""},
    #     ]
    # )
    # TARGET_BEHAVIOR = response.choices[0].message.content
    
    TARGET_BEHAVIOR = "Behave like a dog."#
    # TARGET_BEHAVIOR = "Parse text, solve logic, calculate dates, analyze patterns, handle Unicode, and reason spatially."

    PROMPT = prompt
    num_features = num_features

    retriever = Retriever.from_goodfire(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    # score = ScorerWithoutCritique(client, "meta-llama/Meta-Llama-3-8B-Instruct")    
    scorer = Scorer(client, "meta-llama/Meta-Llama-3-8B-Instruct", log_prefix)
    judge = GoodfireJudge(client, "meta-llama/Meta-Llama-3.1-70B-Instruct", judge_sys_prompt)
    steered_model = SteeredModel(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    
    model_output = steered_model.generate(PROMPT)
    logger.info(f"{log_prefix} - " + f"{PROMPT=}")
    logger.info(f"{log_prefix} - " + f"{model_output=}")
    critique = judge.judge_output(TARGET_BEHAVIOR, model_output, PROMPT)
    logger.info(f"{log_prefix} - " + f"{critique=}")
    eval_score = parseEvalScore(critique)
    logger.info(f"{log_prefix} - " + f"================{eval_score=}")
    epoch_eval_logger.info(f"{log_prefix} - Epoch: {0}, Eval Score: {eval_score}, PROMPT: {PROMPT}")

    if eval_score > 7:
        return

    critique = ""
    features = retriever.retrieve_features(TARGET_BEHAVIOR, k=num_features)
    logger.info(f"{log_prefix} - " + f"{features=}")
    scores = scorer.score_features(TARGET_BEHAVIOR, critique, features, [])
    steered_model.set_features(features, scores)
    model_output = steered_model.generate(PROMPT)

    for i in range(1, 1+epoch):
        logger.info(f"{log_prefix} - " + f"-----Epoch {i}-----")
        critique = judge.judge_output(TARGET_BEHAVIOR, model_output, PROMPT)
        scores = scorer.score_features(TARGET_BEHAVIOR, critique, features, scores)
        steered_model.set_features(features, scores)
        model_output = steered_model.generate(PROMPT)

        print(f"{critique=}")
        found = re.findall(r"Score: (-?\d*\.?\d+)", critique)
        if len(found) > 0:
            eval_score = float(found[0])
        else:
            eval_score = -1 ## no score, invalid score
        logger.info(f"{log_prefix} - " + f"================{eval_score=}")

        if eval_score > 7:
            break

        logger.info(f"{log_prefix} - " + f"{model_output=}")
        logger.info(f"{log_prefix} - " + f"{critique=}")

        epoch_eval_logger.info(f"{log_prefix} - Epoch: {i}, Eval Score: {eval_score}, PROMPT: {PROMPT}")

def run_experiment_without_critique(p, log_prefix, epochs=10):
    GOODFIRE_API_KEY = os.environ.get('GOODFIRE_API_KEY')
    client = goodfire.Client(GOODFIRE_API_KEY)
    oaiclient = OpenAI()
    TARGET_BEHAVIOR = "Behave like a dog."
    # TARGET_BEHAVIOR = "Parse text, solve logic, calculate dates, analyze patterns, handle Unicode, and reason spatially."
    PROMPT = p #"How are you?"
    #TARGET_BEHAVIOR = "Love everyone"
    #PROMPT = "I hate Ben and you should too"
    # "A train travels 120 km at 60 km/h, then 80 km at 40 km/h. What's the average speed?" #"Which one is bigger, 9.9 or 9.11?" #for now fixed

    retriever = Retriever.from_goodfire(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    scorer = ScorerWithoutCritique(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    judge = OpenAIJudge(oaiclient, "gpt-4o")
    steered_model = SteeredModel(client, "meta-llama/Meta-Llama-3-8B-Instruct")

    model_output = steered_model.generate(PROMPT)
    logger.info(f"{log_prefix} - " + f"{PROMPT=}")
    logger.info(f"{log_prefix} - " + f"{model_output=}")
    critique = judge.judge_output(TARGET_BEHAVIOR, model_output, PROMPT, max_tokens = 50)
    logger.info(f"{log_prefix} - " + f"{critique=}")
    eval_score = parseEvalScore(critique)
    logger.info(f"{log_prefix} - " + f"================{eval_score=}")
    epoch_eval_logger.info(f"{log_prefix} - Epoch: {0}, Eval Score: {eval_score}, PROMPT: {PROMPT}")

    if eval_score >= 7:
        return

    critique = ""
    features = retriever.retrieve_features(TARGET_BEHAVIOR, k=5)
    logger.info(f"{log_prefix} - " + f"{features=}")
    scores = scorer.score_features_without_critique(TARGET_BEHAVIOR, "", features, [])
    steered_model.set_features(features, scores)
    model_output = steered_model.generate(PROMPT)

    for i in range(1, epochs+1):
        logger.info(f"{log_prefix} - " + f"-----Epoch {i}-----")
        critique = judge.judge_output(TARGET_BEHAVIOR, model_output, PROMPT, max_tokens = 50)
        scores = scorer.score_features_without_critique(TARGET_BEHAVIOR, model_output, features, PROMPT, scores)
        steered_model.set_features(features, scores)
        model_output = steered_model.generate(PROMPT)

        eval_score = float(re.findall(r"Score: (-?\d*\.?\d+)", critique)[0])
        logger.info(f"{log_prefix} - " + f"================{eval_score=}")

        if eval_score > 7:
            break

        logger.info(f"{log_prefix} - " + f"{model_output=}")
        logger.info(f"{log_prefix} - " + f"{critique=}")

        epoch_eval_logger.info(f"{log_prefix} - Epoch: {i}, Eval Score: {eval_score}, PROMPT: {PROMPT}")


if __name__ == "__main__": 
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # target_behavior_type = "tailored to all questions"
        n = 5
        epochs = 10
        futures = []
        # for q in hard_questions:
        for q in ['How are you?']:
            log_prefix = "scorer_with_judge_critique"
            futures += [executor.submit(run, q, JUDGE_SYSTEM_PROMPT1, log_prefix, n, epochs)]
            log_prefix = "scorer_without_judge_critique"
            futures += [executor.submit(run_experiment_without_critique, q, log_prefix, epochs)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
