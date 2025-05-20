import argparse
import ast
import json
import os
import pprint
import random
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from judgetuning.annotation_dataset import (
    AlpacaEvalDataset,
    AnnotationDataset,
    ArenaHardDataset,
)
from judgetuning.experiment_tracker import ExperimentTracker
from judgetuning.judge.eval_judge import eval_judge
from judgetuning.judge.generate_judge_annotations import generate_judge_annotations
from judgetuning.judge.io.judge_alpaca_eval import JudgeAlpacaEval
from judgetuning.judge.judge_length import JudgeLength
from judgetuning.judge.io.judge_arena_hard import JudgeArenaHard
from judgetuning.judge.io.prompt_utils import PromptType
from judgetuning.judge.judge_with_options.judge_options import JudgeOptions

# import judge_lm
from judgetuning.judge.judge_lm import JudgeLM7B
from judgetuning.judge.judge_pandalm import PandaLM7B

# from judgetuning.judge.judge_pandalm_original import PandaLM7BOrig
# from judgetuning.judge.judge_pandalm_single import PandaLM7BSing
from judgetuning.judge.judge_pandalm_skipper import PandaLM7BSkipper
from judgetuning.judge_paths import path_config_judge, path_result_judge, path_judge
from judgetuning.llm_client import (
    OllamaCompletionClient,
    create_client,
)
import json
import logging
import random
from pathlib import Path

import numpy as np

from judgetuning.annotation_dataset import ArenaHardDataset, AlpacaEvalDataset
from judgetuning.join import common_models

np.random.seed(0)
random.seed(0)


def get_models(annotation_dataset):
    # gets a list of models where the first 27 models are common models between alpaca-eval and arena-hard
    # which are shuffled followed by the remaining models specific of the dataset, also shuffled
    _common_models = common_models()
    _distinct_models = sorted(
        list(set(annotation_dataset.models).difference(_common_models))
    )
    random.shuffle(_common_models)
    random.shuffle(_distinct_models)
    return _common_models + _distinct_models


def most_correlated_instructions(
    df_preference,
    elo_chatbot_arena,
    n_seeds: int | None = 20,
):
    prompt_corr_series = []
    prompts = df_preference.index
    if n_seeds is None:
        dd = df_preference.loc[:, elo_chatbot_arena.index]
        prompts_corr = {}
        for prompt in prompts:
            prompts_corr[prompt] = spearmanr(
                elo_chatbot_arena.loc[dd.columns], dd.loc[[prompt]].values.reshape(-1)
            ).statistic
        prompt_corr_series.append(pd.Series(prompts_corr))
    else:
        for _ in range(n_seeds):
            dd = df_preference.loc[:, elo_chatbot_arena.index].sample(
                axis=1, frac=1.0, replace=True
            )
            # dd = df_preference.loc[:, elo_chatbot_arena.index].sample(
            #     axis=1, n=20, replace=False
            # )
            prompts_corr = {}
            for prompt in prompts:
                prompts_corr[prompt] = spearmanr(
                    elo_chatbot_arena.loc[dd.columns],
                    dd.loc[[prompt]].values.reshape(-1),
                ).statistic
            prompt_corr_series.append(pd.Series(prompts_corr))
    df_prompt_corrs = pd.DataFrame(prompt_corr_series)

    prompt_corr_series = df_prompt_corrs.mean(axis=0).sort_values(ascending=False)
    return prompt_corr_series.index.tolist()

def load_sorted_instruction(
    annotation_dataset: AnnotationDataset, models: list[str]
) -> list[str]:
    # return annotation_dataset.instructions
    return annotation_dataset.instruction_index
    df_preference = annotation_dataset.df_winrate_against_baseline(models=models)
    test_models = common_models()
    val_models = [
        m for m in models if m not in test_models and m in df_preference.columns
    ]
    # not sure why this happens
    val_models = [m for m in models if m in df_preference.columns]

    return most_correlated_instructions(
        df_preference.loc[:, val_models],
        annotation_dataset.chatbot_arena_elo_ratings().loc[val_models],
    )

    # df_preference = annotation_dataset.df_winrate_against_baseline()
    # elo_chatbot_arena = annotation_dataset.chatbot_arena_elo_ratings()
    # models = list(set(elo_chatbot_arena.index).intersection(df_preference.columns))
    # elo_chatbot_arena = elo_chatbot_arena.loc[models]
    # df_preference = df_preference.loc[:, models]
    # # for prompt without model generations, we set the worse possible score
    # df_preference.fillna(1, inplace=True)
    #
    # prompts = df_preference.index.tolist()
    # prompts_corr = {}
    # for prompt in prompts:
    #     prompts_corr[prompt] = spearmanr(
    #         elo_chatbot_arena,
    #         df_preference.loc[[prompt]].mean(axis=0).loc[elo_chatbot_arena.index],
    #     ).statistic
    # prompt_corr_series = pd.Series(prompts_corr).sort_values(ascending=False)
    # return prompt_corr_series.index.tolist()


def generate_sorted_instruction_and_models(annotation_dataset, model_instruction_path):
    models = get_models(annotation_dataset)
    instructions = load_sorted_instruction(annotation_dataset, models)

    with open(model_instruction_path, "w") as f:
        f.write(json.dumps(dict(instructions=instructions, models=models)))


def load_instruction_and_models(annotation_dataset) -> tuple[list[str], list[str]]:
    model_instruction_path = (
        Path(__file__).parent
        / "data"
        / f"models-instructions-{annotation_dataset.name}.json"
    )

    if not model_instruction_path.exists():
        logging.warning(
            f"files containing sorted models or instructions for {annotation_dataset.name} not found, regenerating it"
        )
        generate_sorted_instruction_and_models(
            annotation_dataset, model_instruction_path
        )

    with open(model_instruction_path, "r") as f:
        model_instructions = json.load(f)

    return model_instructions["instructions"], model_instructions["models"]


@dataclass
class EvaluateFidelityArgs:
    model: str
    expid: str = "dummy"
    judge_class: str = "judge-option"
    tensor_parallel_size: int = 1
    dataset: str = "arena-hard"
    split: str = "val"
    max_pred_len: int = 4096
    n_models: int = 2
    n_instructions: int | None = None
    provide_confidence: bool = False
    provide_explanation: bool = False
    provide_example: bool = False
    provide_answer: bool = False
    json_output: bool = True
    score_type: str = "pair"
    temperature: float = 0.0
    n_sample_with_shift: int = 1
    n_sample_without_shift: int = 1
    min_criterion: int = 5  # for lmsys
    dtype: str | None = None
    # bit8: bool = False
    # cpu_offloading: bool = False
    # max_new_token: int = 500
    # sample: bool = True
    max_len_prompt: int = 2048
    # partition_test: int = 0


def parse_args() -> EvaluateFidelityArgs:
    parser = argparse.ArgumentParser(
        prog="Evaluate judge",
    )
    parser.add_argument(
        "--expid",
        help="the experiment id, used to retrieve files later which will be saved under ~/judge-tuning-data/experiments/{expid})",
        required=True,
    )
    parser.add_argument(
        "--model",
        help="the model that is served in vllm (--model)",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--judge_class",
        default="judge-option",
        choices=[
            "judge-option",
            "judge-arena-hard",
            "judge-alpaca-eval",
            "judge-length",
            "judge-lm7b",
            "judge-pandalm",
            # "judge-pandalm-single",
            # "judge-pandalm-original",
            "judge-pandalm-skipper",
        ],
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dataset",
        default="arena-hard",
        choices=["alpaca-eval", "arena-hard", "lmsys", "pandalm", "llmbar"],
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
    )
    parser.add_argument(
        "--max_pred_len",
        help="the maximum length",
        default=4096,
        type=int,
    )
    parser.add_argument(
        "--n_models",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--n_instructions",
        type=int,
    )
    parser.add_argument(
        "--min_criterion",
        default=5,
        type=int,
        help="minimum instruction quality to consider when using lmsys",
    )
    parser.add_argument(
        "--provide_confidence",
        default=0,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--provide_explanation",
        default=0,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--provide_example",
        default=0,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--provide_answer",
        default=0,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--json_output",
        default=0,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--temperature",
        default=0,
        type=float,
    )
    parser.add_argument(
        "--score_type",
        default="pair",
        choices=["preference", "likert", "pair", "best-model-identifier", "multi"],
        type=str,
    )
    parser.add_argument(
        "--n_sample_with_shift",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--n_sample_without_shift",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--dtype",
        type=str,
    )
    parser.add_argument(
        "--max_len_prompt",
        type=int,
        default=2048,
    )

    args = parser.parse_args()
    return EvaluateFidelityArgs(**args.__dict__)


def make_judge(args: EvaluateFidelityArgs):
    match args.judge_class:
        case "judge-option":
            completion_client = make_completion_client(args)
            judge = JudgeOptions(
                completion_client=completion_client,
                prompt_type=PromptType(
                    score_type=args.score_type,
                    provide_confidence=bool(args.provide_confidence),
                    provide_explanation=bool(args.provide_explanation),
                    provide_example=bool(args.provide_example),
                    provide_answer=bool(args.provide_answer),
                    json_output=bool(args.json_output),
                    prompt_filename="request_prompt_ours",
                ),
                n_sample_with_shift=args.n_sample_with_shift,
                n_sample_without_shift=args.n_sample_without_shift,
                temperature_model=args.temperature,
            )
        case "judge-arena-hard":
            completion_client = make_completion_client(args)
            judge = JudgeArenaHard(
                completion_client=completion_client,
            )
        case "judge-alpaca-eval":
            completion_client = make_completion_client(args)
            judge = JudgeAlpacaEval(
                completion_client=completion_client,
            )
        case "judge-length":
            judge = JudgeLength()
        case "judge-lm7b":
            import torch
            judge = JudgeLM7B(
                device="cuda" if torch.cuda.is_available() else "cpu",
                # load_8bit=bool(args.bit8),
                # cpu_offloading=bool(args.cpu_offloading),
                # temperature=args.temperature,
                # max_new_token=args.max_pred_len,
            )
        case "judge-pandalm":
            judge = PandaLM7B(
                # temperature=args.temperature,
                # max_new_tokens=args.max_pred_len,
                max_len_prompt=args.max_len_prompt,
            )
        # case "judge-pandalm-single":
        #     judge = PandaLM7BSing(
        #         # temperature=args.temperature,
        #         # max_new_tokens=args.max_pred_len,
        #     )
        # case "judge-pandalm-original":
        #     judge = PandaLM7BOrig(
        #         # temperature=args.temperature,
        #         # max_new_tokens=args.max_pred_len,
        #     )
        case "judge-pandalm-skipper":
            judge = PandaLM7BSkipper(
                # temperature=args.temperature,
                # max_new_tokens=args.max_pred_len,
                max_len_prompt=args.max_len_prompt,
            )
    return judge


def make_completion_client(args: EvaluateFidelityArgs):
    try:
        completion_client = create_client(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_pred_len=args.max_pred_len,
            dtype=args.dtype,
        )
    except ImportError as e:
        print(str(e))
        args.model = "Ollama"
        print(
            "cannot import vllm client, assuming you are running on a machine without GPU and using Ollama"
        )
        completion_client = OllamaCompletionClient()
    return completion_client


def main():
    np.random.seed(0)
    random.seed(0)
    args = parse_args()

    print(f"Evaluation args: {args}")
    try:
        import torch

        print(f"GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    except Exception as e:
        print(e)
    jobname = os.getenv("SP_JOBNAME", "")
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        jobname += "-" + os.getenv("SLURM_ARRAY_TASK_ID")

    n_models = args.n_models
    n_instructions = args.n_instructions
    expid = args.expid

    match args.dataset:
        case "arena-hard":
            annotation_dataset = ArenaHardDataset(
                keep_only_chatbot_arena=True, rename_chatbot_arena=True
            )
            baseline_model = "gpt-4-0314"
        case "alpaca-eval":
            annotation_dataset = AlpacaEvalDataset(
                keep_only_chatbot_arena=True, rename_chatbot_arena=True
            )
            baseline_model = "gpt-4-0314"
        case _:
            raise ValueError(f"Invalid dataset: {args.dataset}")
    print(f"Loaded dataset: {annotation_dataset.name}.")
    instructions, models = load_instruction_and_models(annotation_dataset)
    if args.dataset == "arena-hard":
        # use model available for claude and GPT4 annotation in arena-hard
        models = [
            "gpt-4-turbo-2024-04-09",
            "gpt-4-0125-preview",
            "claude-2.1",
            "llama-2-70b-chat",
            "gpt-3.5-turbo-1106",
            "gemma-2b-it",
            "yi-34b-chat",
            "dbrx-instruct-preview",
            "qwen1.5-72b-chat",
            "gpt-4-1106-preview",
            "gemma-7b-it",
            "claude-2.0",
            "mistral-medium",
            "claude-3-sonnet-20240229",
            "mixtral-8x7b-instruct-v0.1",
            "gemini-pro",
            "tulu-2-dpo-70b",
            "gpt-4-0613",
            "gpt-3.5-turbo-0613",
            "mistral-large-2402",
        ]

    if baseline_model in models:
        # no need to judge the baseline with itself
        models.remove(baseline_model)

    judge = make_judge(args)
    print(f"Generating annotations for judge {judge}")

    subset_instructions = instructions[:n_instructions]
    subset_models = models[:n_models]

    # judge_name = "_".join([str(x).replace("/", "-") for x in args.__dict__.values()])
    print(
        f"Going to evaluate {n_instructions} instructions on {n_models} models for job {jobname} ({judge}).\n"
    )
    print(f"Models considered: {subset_models}")

    experiment_tracker = ExperimentTracker(
        judge=judge,
        instructions=subset_instructions,
        instruction_dataset=args.dataset,
        models=subset_models,
        jobname=jobname,
        tags=[expid],
    )
    # save judge configuration
    path_config_judge(expid=expid, judge_name=jobname).parent.mkdir(
        parents=True, exist_ok=True
    )
    with open(path_config_judge(expid=expid, judge_name=jobname), "w") as f:
        f.write(judge.to_json())
    with open(path_judge(expid=expid, judge_name=jobname) / "args.json", "w") as f:
        f.write(json.dumps(args.__dict__))

    generate_judge_annotations(
        annotation_dataset=annotation_dataset,
        expid=expid,
        judge=judge,
        judge_name=jobname,
        instructions_index=subset_instructions,
        models=subset_models,
        experiment_tracker=experiment_tracker,
        baseline_model=baseline_model,
    )
    metrics = eval_judge(
        expid=expid,
        judge_name=jobname,
        annotation_dataset=annotation_dataset,
    )

    print(f"Results for {judge}:")
    pprint.pprint(metrics, indent=2)

    # save results into json file
    with open(path_result_judge(expid=expid, judge_name=jobname), "w") as f:
        f.write(json.dumps(metrics.__dict__))

    experiment_tracker.track_results(metrics)


if __name__ == "__main__":
    main()
