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

def run(prompt, target_behavior_type, num_features=5):
    GOODFIRE_API_KEY = os.environ.get('GOODFIRE_API_KEY')
    client = goodfire.Client(GOODFIRE_API_KEY)
    oaiclient = OpenAI()
    # TARGET_BEHAVIOR = "Correctly answer the questions in the prompt."
    TARGET_BEHAVIOR = "Parse text, solve logic, calculate dates, analyze patterns, handle Unicode, and reason spatially."
    # target_behavior_type = "tailored to all questions"
    PROMPT = prompt
    num_features = num_features

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
    epoch_eval_logger.info(f"Epoch: {0}, Eval Score: {eval_score}, PROMPT: {PROMPT}, TARGET_BEHAVIOR: {target_behavior_type}, num_features = {num_features}")

    if eval_score > 7:
        return

    critique = ""
    features = retriever.retrieve_features(TARGET_BEHAVIOR, k=num_features)
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

        print(f"{critique=}")
        found = re.findall(r"Score: (-?\d*\.?\d+)", critique)
        if len(found) > 0:
            eval_score = float(found[0])
        else:
            eval_score = -1 ## error
        logging.info(f"================{eval_score=}")

        if eval_score > 7:
            break

        logging.info(f"{model_output=}")
        logging.info(f"{critique=}")

        epoch_eval_logger.info(f"Epoch: {i}, Eval Score: {eval_score}, PROMPT: {PROMPT}, TARGET_BEHAVIOR: {target_behavior_type}, num_features = {num_features}")


if __name__ == "__main__":   
    # parse epoch_eval_logs.txt to a dict where key is the num_features and value is a dict of key=target_behavior_type and value=dict of key=prompt and value=(epoch, eval_score)
    epoch_eval_dict = {}
    with open('epoch_eval_logs.txt', 'r') as file:
        log_entries = file.readlines()
        for entry in log_entries:
            match = re.search(r"Epoch: (\d+), Eval Score: ([\d.]+), PROMPT: (.*?), TARGET_BEHAVIOR: (.*?), num_features = (\d+)", entry)
            if match:
                epoch = int(match.group(1))
                eval_score = float(match.group(2))
                prompt = match.group(3)
                target_behavior = match.group(4)
                num_features = int(match.group(5))
                if num_features not in epoch_eval_dict:
                    epoch_eval_dict[num_features] = {}
                if target_behavior not in epoch_eval_dict[num_features]:
                    epoch_eval_dict[num_features][target_behavior] = {}
                epoch_eval_dict[num_features][target_behavior][prompt] = (epoch, eval_score)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        target_behavior_type = "tailored to all questions"
        for n in [10, 15]:
            for q in questions_dict:
            # check epoch_eval_dict to see if the num_features, target_behavior, prompt combo has been evaluated
                if n in epoch_eval_dict.keys():
                    if target_behavior_type in epoch_eval_dict[n].keys():
                        if q in epoch_eval_dict[n][target_behavior_type]:
                            continue
                futures = [executor.submit(run, q, target_behavior_type, n) for q in hard_questions]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
