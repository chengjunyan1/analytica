from lllm.models import Prompt
from pydantic import BaseModel
from analytica.agent.prompts.prover import concluding_parser

# set a default path for all prompts


BACKGROUND = '''
You are an expert in analysis.
You will be provided with a task, which contains a question and a list of options. Your task is to analyze and evaluate the options.
Every time, you will be asked to analyze one option only. For each option, you will need to provide one analysis report that elaborates your thoughts and reasoning, then provide a rating of this option.
'''

TASK_DESCRIPTION = '''
The current date is {date}. Please analyze the following task:

{task}
'''




class TaskResponse(BaseModel):
    report: str
    rating: float


main_system_prompt = Prompt(
    path='system',
    prompt=BACKGROUND
)



task_query_prompt = Prompt(
    path='task_query',
    prompt=TASK_DESCRIPTION + '''

Please analyze the option {option_idx}: {option_description}. 
Provide an analysis report first that elaborates your thoughts and reasoning,
then provide a float number rating from 0.00 to 1.00 for this option.
    ''',
    format=TaskResponse,
    allow_web_search=True,
)






####### Vanilla Prover #######
     


VANILLA_PROVER_BACKGROUND = '''
You are an expert in analysis.
You will be provided with a proposition, and your task is to provide a comprehensive proof that either proves or disproves the proposition. 
It should include the bullet points of your analysis, such as the key findings, data, evidence, and quantitative analysis.
Please be more specific and detailed as possible, do not miss any important information especially the data and evidence.
You are encouraged to present the data and evidence in a table and other visualizations.

### Iterative Refinement Process (Mental Sandbox)
Before finalizing your JSON, you MUST perform an iterative "sanity check":
1. **Base Rate Check**: Is this claim extraordinary or unlikely a priori?
    - *Example*: "Company X will go bankrupt" has a low base rate (<0.05). Start there. Only move up if evidence is strong.
2. **Draft Probability**: Based on the evidence, what is your initial gut estimate for `p_true`?
3. **Check Calibration**: Look at the table below. Does your estimate match the definition?
    - *Example*: You estimated 0.80, but the evidence has significant gaps. The table says 0.70-0.89 requires "consistent data". Should you lower it to 0.65 (Likely)?
4. **Refine**: Adjust your probability to be consistent with the strict definitions in the table.

After you have written the complete textual proof, append a single ```json ... ``` block. Inside this block, provide a single JSON object with exactly two keys:
1.  "p_true": Your estimated probability (a float between 0.00 and 1.00) that the proposition is true.
    **Probability Calibration Table:**
    - **1.00 (Certainty)**: Proven True by direct, irrefutable facts.
    - **0.90 - 0.99 (Extremely Likely)**: Overwhelming evidence, virtual certainty.
    - **0.70 - 0.89 (Very Likely)**: Strong evidence, consistent data.
    - **0.55 - 0.69 (Likely)**: Balance of evidence tilts positive, but uncertainties remain.
    - **0.50 (Unknown)**: No information or perfectly balanced evidence.
    - **0.31 - 0.45 (Unlikely)**: Balance of evidence tilts negative.
    - **0.11 - 0.30 (Very Unlikely)**: Strong counter-evidence.
    - **0.01 - 0.10 (Extremely Unlikely)**: Overwhelming counter-evidence.
    - **0.00 (Impossibility)**: Proven False.
    *Note: Use precise values (e.g., 0.72, 0.15) to reflect specific nuances in evidence strength.*
2.  "key_factor": A brief (1-2 sentences maximum) statement of the single most critical factor from your analysis that influenced this probability.

Example of the final part of your response:
... (end of your textual proof) ...
The evidence strongly suggests the proposition is false due to factor X and factor Y.

```json
{
  "p_true": 0.12,
  "key_factor": "The consistent downtrend in the primary dataset combined with negative macroeconomic indicators."
}
```
'''



vanilla_prover_system_prompt = Prompt(
    path='vanilla_prover_system',
    prompt=VANILLA_PROVER_BACKGROUND
)





vanilla_prover_task_prompt = Prompt(
    path='vanilla_prover_task',
    prompt='''
Here is the proposition you are going to analyze:
{proposition}

**Data Timeliness**: Remember the current date is {date}. 

Now please provide your analysis and proof. Remember to provide the final part of your response in the format of ```json ... ``` block.
''',
    md_tags=['json'],
    required_md_tags=['json'],
    parser=concluding_parser,
    allow_web_search=True,
)





