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
from datetime import datetime
import logging
import threading

def parseEvalScore(critique: str):
    return float(re.findall(r"Score: (-?\d*\.?\d+)", critique)[0])

# Configure logging
logging.basicConfig(filename='scorer_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
import concurrent.futures

# Create a thread-safe logging handler
class ConcurrentLogHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self._handler = logging.FileHandler(filename)
        self._handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self._lock = threading.Lock()

    def emit(self, record):
        with self._lock:
            self._handler.emit(record)

# Configure concurrent logging for main logs
concurrent_log_handler = ConcurrentLogHandler('scorer_logs.txt')
logging.getLogger().addHandler(concurrent_log_handler)

# Configure shared epoch evaluation logger
epoch_eval_handler = ConcurrentLogHandler('epoch_eval_logs.txt')
epoch_eval_logger = logging.getLogger('epoch_eval')
epoch_eval_logger.setLevel(logging.INFO)
epoch_eval_logger.addHandler(epoch_eval_handler)

def run(prompt):
    GOODFIRE_API_KEY = os.environ.get('GOODFIRE_API_KEY')
    client = goodfire.Client(GOODFIRE_API_KEY)
    oaiclient = OpenAI()
    TARGET_BEHAVIOR = "Correctly answer the questions in the prompt."
    PROMPT = prompt

    retriever = Retriever.from_goodfire(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    scorer = Scorer(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    judge = OpenAIJudge(oaiclient, "gpt-4o")
    steered_model = SteeredModel(client, "meta-llama/Meta-Llama-3-8B-Instruct")
    
    model_output = steered_model.generate(PROMPT)
    logging.info(f"{PROMPT=}")
    logging.info(f"{model_output=}")
    critique = judge.judge_output(TARGET_BEHAVIOR, model_output, PROMPT)
    logging.info(f"{critique=}")
    eval_score = parseEvalScore(critique)
    logging.info(f"================{eval_score=}")
    epoch_eval_logger.info(f"Epoch: {0}, Eval Score: {eval_score}, PROMPT: {PROMPT}, TARGET_BEHAVIOR: {TARGET_BEHAVIOR}")

    if eval_score > 7:
        return

    critique = ""
    features = retriever.retrieve_features(TARGET_BEHAVIOR, k=5)
    logging.info(f"{features=}")
    scores = scorer.score_features(TARGET_BEHAVIOR, critique, features, [])
    steered_model.set_features(features, scores)
    model_output = steered_model.generate(PROMPT)

    for i in range(1, 21):
        logging.info(f"-----Epoch {i}-----")
        critique = judge.judge_output(TARGET_BEHAVIOR, model_output, PROMPT)
        scores = scorer.score_features(TARGET_BEHAVIOR, critique, features, scores)
        steered_model.set_features(features, scores)
        model_output = steered_model.generate(PROMPT)

        eval_score = float(re.findall(r"Score: (-?\d*\.?\d+)", critique)[0])
        logging.info(f"================{eval_score=}")

        if eval_score > 7:
            break

        logging.info(f"{model_output=}")
        logging.info(f"{critique=}")

        epoch_eval_logger.info(f"Epoch: {i}, Eval Score: {eval_score}, PROMPT: {PROMPT}, TARGET_BEHAVIOR: {TARGET_BEHAVIOR}")


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run, q) for q in hard_questions]
        for future in concurrent.futures.as_completed(futures):
            future.result()
