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
from scorer import Scorer
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

def run(prompt, judge_sys_prompt=JUDGE_SYSTEM_PROMPT, log_prefix="", num_features=5):
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
    
    TARGET_BEHAVIOR = "Parse text, solve logic, calculate dates, analyze patterns, handle Unicode, and reason spatially."

    PROMPT = prompt
    num_features = num_features

    retriever = Retriever.from_goodfire(client, "meta-llama/Meta-Llama-3-8B-Instruct")
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
    epoch_eval_logger.info(f"Epoch: {0}, Eval Score: {eval_score}, PROMPT: {PROMPT}, JUDGE_SYSTEM_PROMPT_TYPE: {log_prefix}, num_features = {num_features}")

    if eval_score > 7:
        return

    critique = ""
    features = retriever.retrieve_features(TARGET_BEHAVIOR, k=num_features)
    logger.info(f"{log_prefix} - " + f"{features=}")
    scores = scorer.score_features(TARGET_BEHAVIOR, critique, features, [])
    steered_model.set_features(features, scores)
    model_output = steered_model.generate(PROMPT)

    for i in range(1, 6):
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

        epoch_eval_logger.info(f"Epoch: {i}, Eval Score: {eval_score}, PROMPT: {PROMPT}, JUDGE_SYSTEM_PROMPT_TYPE: {log_prefix}, num_features = {num_features}")


if __name__ == "__main__":   
    with concurrent.futures.ThreadPoolExecutor() as executor:
        target_behavior_type = "tailored to all questions"
        n = 5
        futures = []
        for i, s in enumerate([JUDGE_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT1]):
            futures += [executor.submit(run, q, s, f"JUDGE_SYSTEM_PROMPT{i}", 5) for q in hard_questions]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    # # synchoronous
    # for i, s in enumerate([JUDGE_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT1]):
    #     for q in hard_questions:
    #         run(q, s, f"JUDGE_SYSTEM_PROMPT{i}", 5)
