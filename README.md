# Analytica

This is the official repo for the ICLR 2026 paper "Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis". Demo can be found here at https://analyt1.com/. Analytica is built based on the LLLM framework (https://github.com/chengjunyan1/lllm), check lllm.one for details.


## Installation

```bash
conda create -n analytica python=3.13 -y &&\
conda activate analytica &&\
cd ~/analytica &&\
pip install -e . &&\
analytica setup &&\
pip install -r requirements.txt &&\
python -m ipykernel install --user --name "analytica" --display-name "Python (analytica)"
```

## Set up Environment Variables

```bash
export MY_OPENAI_KEY=YOUR_OPENAI_KEY
export FMP_API_KEY=YOUR_FMP_API_KEY
export EXA_API_KEY=YOUR_EXA_API_KEY
export FRED_API_KEY=YOUR_FRED_API_KEY
export KALSHI_API_KEY_ID=YOUR_KALSHI_API_KEY_ID
export SEARCH_API_KEY=YOUR_SEARCH_API_KEY
export PM_PRIVATE_KEY=YOUR_PM_PRIVATE_KEY # Polymarket 
export CACHE_DIR=YOUR_CACHE_DIR # e.g., ~/analytica/.cache
export DB_DIR=YOUR_DB_DIR # e.g., ~/analytica/analytica/db
export TMP_DIR=YOUR_TMP_DIR # e.g., ~/analytica/.tmp
export CKPT_DIR=YOUR_CKPT_DIR # e.g., ~/analytica/ckpt
export LOG_DIR=YOUR_LOG_DIR # e.g., ~/analytica/.log
```

Optional Environment Variables:

```bash
export GOLDEN_API_KEY=YOUR_GOLDEN_API_KEY # Golden KG
export MSD_API_KEY=YOUR_MSD_API_KEY # Main Street Data
export WA_API_DEV=YOUR_WA_API_DEV # Wolfram Alpha API
export WA_API_KEY=YOUR_WA_API_KEY # Wolfram Alpha API
```


## Run


### Simple query

```python
import analytica.utils as U
from analytica.system import build_system

# put your own config file path
config = U.load_config(f'configs/test.yaml')
system = build_system(config)

from analytica.agent.ae import Proposition
proposition = Proposition(
    sentence="There will be rain in San Francisco today.",
    context="",
    date=U.dt_now_str(),
    ckpt_dir="ckpt/test",
)

# must be an analytica or grounder/prover agent
proposition = system.agent.prove(proposition) # note, may need to await if agent is async
print(proposition.json)
```


### Run evaluation

```python
import analytica.utils as U
from analytica.system import build_system

config = U.load_config(f'configs/test.yaml')
system = build_system(config)
if system.agent.is_async:
    await system.async_evaluate(max_concurrent=10)
else:
    system.evaluate(max_concurrent=10)
```

The results will be saved in the ckpt directory.


## Cite

```bibtex
@inproceedings{
cheng2026analytica,
title={Analytica: Soft Propositional Reasoning for Robust and Scalable {LLM}-Driven Analysis},
author={Junyan Cheng and Kyle Richardson and Peter Chin},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=9cFT6u82uh}
}
```

