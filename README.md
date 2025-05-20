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
PYTHONPATH=. uv run python results_scripts/figure1.py 
```

or with pip:
```
pip install -r requirements.txt
PYTHONPATH=. python results_scripts/figure1.py 
```

## Evaluating your own judge

TODO Omar.

## Evaluating judge configurations

TODO David.

## Evaluating baselines

TODO Omar.

## Generating figures & results

See `[results_scripts](results_scripts)`, for instance:
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