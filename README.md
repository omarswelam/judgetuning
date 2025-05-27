# Judge Tuning

This repository allows to tune judge configurations and reproduce the results of [this ICML 2025 paper](https://arxiv.org/abs/2501.17178).

## How to install

First download the repository:
```bash
git clone https://github.com/geoalgo/judgetuning
cd judgetuning
```

Then install with uv (recommended):
```
uv sync .
uv add -r requirements.txt
PYTHONPATH=. python results_scripts/figure1.py 
```

or with pip:
```
pip install -r requirements.txt
PYTHONPATH=. python results_scripts/figure1.py 
```

## Evaluating a judge


### Evaluating your own judge

To evaluate your own judge, you can define the `JudgeCustom` object here [class](judgetuning/judge/judge_custom.py). This will require you to define:
- `preference`: the function you use for judging which output is preferred.
- `swap`: flag for indicating the positions of the outputs were swapped for future analysis of position bias.
- `judge_completion`: the text completion from an LLM judge used for evaluating `preference`.

To evaluate on our datasets, define your function in the `judge-custom` case in the `make_judge` function [here](judgetuning/script/evaluate_spearman_correlation.py).


### Reevaluating a given judge configurations

If you want to reevaluate one of the 4480 judge configuration, you can run

```
PYTHONPATH=. python judgetuning/script/evaluate_human_agreement.py --expid test --judge_class judge-length --dataset pandalm --split test
PYTHONPATH=. python judgetuning/script/evaluate_human_agreement.py --expid test --judge_class judge-option --dataset lmsys \
--split val --provide_confidence 1 --provide_example 0 --json_output 1 --temperature 0.001 --score_type likert 
```

See `parse_args` in `evaluate_spearman_correlation.py` for 
other supported options, for instance dataset can be "lmsys", "pandalm", "llmbar" for `evaluate_human_agreement.py`, 
you can configure the judge class to be our tunable class, arena-hard, alpaca-eval, pandalm, judgelm, etc.

To evaluate Spearman correlation on chatbot arena, you can run:

```
PYTHONPATH=. python judgetuning/script/evaluate_spearman_correlation.py  --expid test --judge_class judge-length --dataset alpaca-eval --split test 
```
As above, you can customize the judge and other options, see `parse_args` to get the list of supported options.


### Evaluating baselines

To regenerate the PandaLM and JudgeLM, you need to create the datasets and then run the human agreement score. 

To generate the datasets, run the following:
- for the llmsys datasets: `python judgetuning/annotation_dataset/tables/tables_lmsys_kaggle.py`
- for the pandalm datasets: `python judgetuning/annotation_dataset/tables/tables_pandalm.py`

To regenerate the results, run the following:
```bash
METHOD=judge-lm7b  # can also be judge-pandalm
DATASET=lmsys 
# evaluate judgelm on human agreement
python judgetuning/script/evaluate_human_agreement.py --judge_class=$METHOD --max_len_prompt=8192 --max_pred_len=1024 --split=test --dataset=$DATASET --expid=random_run
```
- for evaluating JudgeLM on LLMSyS, set `METHOD` to `judge-lm7b`
- for evaluating PandaLM on LLMSyS, set `METHOD` to `judge-pandalm`
- for evaluating PandaLM on PandaLM dataset, set `METHOD` to `judge-pandalm` and `DATASET` to `pandalm`

## Generating figures & results

See [results_scripts](results_scripts/), for instance:
```bash
python results_scripts/figure1.py
python results_scripts/table2.py
```

where you can replace by the name of figure/table needed.


## Citation

If you use this work in your research, please cite the following paper:

```bash
@misc{salinas2025tuningllmjudgedesign,
      title={Tuning LLM Judge Design Decisions for 1/1000 of the Cost}, 
      author={David Salinas and Omar Swelam and Frank Hutter},
      year={2025},
      eprint={2501.17178},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.17178}, 
}
```

TODO update with ICML bibtex.
