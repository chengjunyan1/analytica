from lllm.models import Prompt, Function, FunctionCall
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List
from typing import Dict, Any
# set a default path for all prompts
from lllm.utils import find_all_xml_tags_sorted
from lllm.models import ParseError, default_parser
from lllm.llm import TRUE_TOKEN, FALSE_TOKEN
import json

class TaskResponse(BaseModel):
    report: str
    rating: float


# One session one proposition, multiple programs in a sandbox session with single view provided



###########################################
#  System prompt
###########################################


# You will be provided with a task, which contains a question and a list of options. Your task is to analyze and evaluate the options.
# Every time, you will be asked to analyze one option only. For each option, you will need to provide one analysis report that elaborates your thoughts and reasoning, then provide a rating of this option.







_PROGRAMMING_CORE_SYSTEM = '''
You are an expert in analysis.
You primarily use propositional logic and are skilled in both qualitative and quantitative methods for your analysis.
Your primary goal is to prove or disprove the given proposition.
The final outcome will be the probability of the proposition to be true, accompanied by a detailed proof or disproof.
You use a Jupyter Notebook environment as your analysis tool.
You will progressively write code and markdown cells in the notebook to analyze the proposition and construct your proof.

Please read these instructions carefully.

## Jupyter Notebook Environment

You will work by generating content for a Jupyter Notebook.
Every time you respond, you will provide one or more cells.

1.  **Python Cells**: Wrap Python code in `<python_cell> </python_cell>` tags. Use these for quantitative analysis, data processing, API calls, visualizations (using matplotlib or plotly, avoid altair due to rendering issues), etc.
2.  **Markdown Cells**: Wrap markdown content in `<markdown_cell> </markdown_cell>` tags. Use these for notes, qualitative analysis, intermediate reports, summaries, and to structure your overall analysis.
3.  **Cell Order**: Cells are added to the notebook in the order you provide them.
4.  **Sequential Execution**: Cells are executed sequentially.
5.  **Error Handling**: If a Python cell execution fails, you will be informed of the error and required to provide a corrected version of *that specific cell*. The notebook will then re-run from the corrected cell.
6.  **Immutability**: You cannot delete or edit previously submitted cells. Each response appends new cells.
7.  **Output Availability**: You will only receive the outputs of the cells you wrote in the *current* response. Outputs from previous turns are part of the dialog history.


## API Library Usage

You have access to an API library within your `<python_cell>` blocks. The system will execute these for you:

 **`CALL_API(api_path: str, api_params: dict)`**: Use this to call an API endpoint.
    * Example: `response = CALL_API("fmp/crypto/end-of-day/historical-price-eod/full", {{"symbol": "BTCUSD", "from": "2023-01-01"}}) `

A directory of available APIs is provided below.
1.  **Consult Documentation First**: ALWAYS make sure you have retrieved and read the documents of the API endpoints you 
are going to use *before* you write a Python cell that uses `CALL_API` function, unless you have retrieved that specific
documentation earlier in this session. This prevents incorrect API usage. 
2.  **Use `CALL_API`**: ALWAYS use the `CALL_API` function to interact with APIs. API keys are managed by the backend. 

## Terminating Analysis

When your analysis is complete and you are ready to construct your final proof, use the following instruction *by itself* in your response (do not include any `<python_cell>` in that same response):

```
<TERMINATE_NOTEBOOK>
```
You will then be prompted to provide your final proof and conclusion.

## API Directory

---
```
{api_directory}
```
---

Additional Notes:
* Avoid rendering libraries like Altair due to potential display issues. Matplotlib or Plotly are preferred for visualizations.
* Do not repeatedly request the same API documentation if you've already retrieved it.
* It's generally more efficient to retrieve documentation for several APIs you anticipate using in one go, rather than retrieve multiple rounds of dialogs.
'''




###########################################
#  Programming Core Prompt
###########################################



# Put as much fixed prompts as possible in the system prompt
prover_system_prompt = Prompt(
    path='system',
    prompt=(
        _PROGRAMMING_CORE_SYSTEM 
    )
)



prover_request_prompt = Prompt(
    path='request',
    prompt='''
Here is the proposition you are going to analyze:
{proposition}

**Data Timeliness**: Remember the current date is {date}. APIs might return data that is newer, but your analysis should be grounded in information available up to this date. Treat any data post-{date} with caution or as future projections if explicitly stated by the API.
'''
)


def retrieve_api_doc_processor(result: str, function_call: FunctionCall):
    return f'''Here are the documentations for the APIs you requested for {function_call.arguments['full_paths']}:
---
{result}
---
'''

retrieve_api_doc_func = Function(
    name='retrieve_api_doc',
    description="Retrieve the API documentation for the given API full paths from the directory.",
    properties={
        'full_paths': {
            'type': 'array',
            'items': {'type': 'string'},
            'minItems': 1,
            'maxItems': 25,
            'description': 'A list of API full paths to retrieve the documentation for. You can request documentation for multiple API paths in a single request. It must be the full path of the API from the directory.'
        }
    },
    required=['full_paths'],
    processor=retrieve_api_doc_processor,
    allow_web_search=True,
)



programming_interrupt_prompt = '''
{call_results}


Based on the results, please:
 * Continue retrieving more API docs if you are not satisfied with the results.
 * If there are error message from the API backend (such as the wrong API path), you may wish to retry.
 * If you think you got sufficient information, you can start to provide your analysis and writing the notebook cells.

Remember to follow all previously stated guidelines.
'''



def programming_parser(message: str, **kwargs):
    errors = []
    warnings = []
    matches = find_all_xml_tags_sorted(message)
    if len(matches) == 0:
        errors.append(f"No python or markdown cells found, it should be provided as <python_cell>...</python_cell> or <markdown_cell>...</markdown_cell>, or <TERMINATE_NOTEBOOK> to terminate the notebook.")
    terminate_notebook = '<TERMINATE_NOTEBOOK>' in message
    cells = []
    n_python_cells = 0
    for match in matches:
        if match['tag'] in ['python_cell', 'markdown_cell']:
            cells.append((match['tag'], match['content']))
            if match['tag'] == 'python_cell':
                n_python_cells += 1
        else:
            errors.append(f"Invalid xml tag: {match['tag']} (expected: python_cell, markdown_cell)")
    if len(cells) == 0 and not terminate_notebook:
        errors.append(f"No cells found, it should be provided as <python_cell>...</python_cell> or <markdown_cell>...</markdown_cell>")
    if len(errors) > 0:
        raise ParseError(f"Parsing errors:\n{'\n'.join(errors)}")
    if terminate_notebook and n_python_cells > 0:
        warnings.append(f'Detected both <TERMINATE_NOTEBOOK> and python cells, please provide only one of them. <TERMINATE_NOTEBOOK> will be ignored.')
        terminate_notebook = False
    parsed = {
        'raw': message,
        'cells': cells,
        'terminate_notebook': terminate_notebook,
        'warnings': warnings
    }
    return parsed

def initial_programming_parser(message: str, **kwargs):
    parsed = programming_parser(message, **kwargs)
    if parsed['terminate_notebook']:
        if len(parsed['cells']) > 0:
            parsed['warnings'].append(f'Terminating the notebook is not allowed in the first step, your <TERMINATE_NOTEBOOK> will be ignored.')
        else:
            raise ParseError(f'Terminating the notebook is not allowed in the first step, your <TERMINATE_NOTEBOOK> will be ignored. Please write cells.')
    return parsed




COMMON_PROGRAMMING_INSTRUCTIONS = '''
You are encouraged to perform reasoning and analysis process in your response before or in between you write cells.
Wrap Python code cells in `<python_cell> ... </python_cell>` tags.
Wrap markdown cell content in `<markdown_cell> ... </markdown_cell>` tags.
Remember to close each cell tag correctly.
Use the `CALL_API()` within Python cells to assist your analysis.
Make sure to read the documentation for the APIs you are using before using them.
And only use `CALL_API` function to call the APIs.
'''


initial_programming_prompt = Prompt(
    path='initial_programming',
    prompt=f'''Please start your analysis using the Jupiter Notebook.
{COMMON_PROGRAMMING_INSTRUCTIONS}
You are not allowed to terminate the notebook in the first response as it is empty.
''',
    xml_tags=['python_cell', 'markdown_cell'],
    signal_tags=['TERMINATE_NOTEBOOK'],
    _functions=[retrieve_api_doc_func],
    parser=initial_programming_parser,
    interrupt_prompt=programming_interrupt_prompt,
    allow_web_search=True,
) 




next_programming_prompt = Prompt(
    path='next_programming',
    prompt=f'''
Please continue your analysis by writing the notebook cells.
{COMMON_PROGRAMMING_INSTRUCTIONS}
If you wish to terminate the notebook, please provide a <TERMINATE_NOTEBOOK> instruction.
And remember to not include any python cells in your response when you provide a <TERMINATE_NOTEBOOK> instruction, otherwise, your termination instruction will be IGNORED.
''',
    xml_tags=['python_cell', 'markdown_cell'],
    signal_tags=['TERMINATE_NOTEBOOK'],
    _functions=[retrieve_api_doc_func],
    parser=programming_parser,
    interrupt_prompt=programming_interrupt_prompt,
    allow_web_search=True,
)





prove_debugging_prompt = Prompt(
    path='debugging',
    prompt='''Encountered an error during the execution of one of your Python cells. The error information is as follows:

---
{error_info}
---

Please fix the error by providing an updated version of the *erroneous Python cell only*.
You are encouraged to perform some reasoning and analysis process of the error before providing the updated code.
Wrap your updated code with <python_cell> ... </python_cell> tags in your response.
DO NOT forget to close the python code block with </python_cell> tags.
You MUST provide one and only one python code cell in your response.
If multiple `<python_cell>` tags are provided, only the LAST one will be considered as the fix for the erroneous cell.
''',
    xml_tags=['python_cell'],
    required_xml_tags=['python_cell'],
    allow_web_search=True,
)

warnings_message_prompt = Prompt(
    path='warnings_message',
    prompt='''Here are the warnings from the system regarding the previous response:
---
{warnings}
---
''',
    allow_web_search=True,
)


function_calls_message_prompt = Prompt(
    path='function_calls_message',
    prompt='''Here is a record of the function calls you made when generating the response:

---
{function_calls}
---
''',
    allow_web_search=True,
)

debugging_truncate_prompt = Prompt(
    path='debugging_truncate',
    prompt='''Encountered error when executing the notebook. The error information is as follows:

---
{error_info}
---

The errors have been fixed. The cells in your response have been updated as follows when trying to fix the errors:

---
{program_updates}
---
''',
    xml_tags=['debugging_summary'],
    allow_web_search=True,
)


###########################################
#  Conclude prompts
###########################################


max_step_reached_prompt = Prompt(
    path='max_step_reached',
    prompt='You have reached the maximum number of steps if analysis. The notebook is terminated.',
    allow_web_search=True,
)



# concluding_prompt = Prompt(
#     path='concluding',
#     prompt='Please conclude your analysis by providing a proof that either proves or disproves the proposition based on your analysis in the notebook.',
# )


# classify_prompt = Prompt(
#     path='classify',
#     prompt='''Now, please provide the final conclusion of your analysis based on the notebook and your proof.
# Your response MUST be a single word: either "{TRUE_TOKEN}" or "{FALSE_TOKEN}", indicating the validity of the proposition.
# Do not provide any other text, explanation, or formatting.
# ''',
# )



def concluding_parser(message: str, **kwargs):
    parsed = default_parser(message, md_tags=['json'], required_md_tags=['json'])
    json_blocks = parsed['md_tags']['json']
    if len(json_blocks) != 1:
        raise ParseError("Please provide one and only one JSON block")
    conclusion = json_blocks[0]
    if isinstance(conclusion, list):
        if len(conclusion) != 1:
            raise ParseError("Please provide one and only one conclusion")
        conclusion = conclusion[0]
    try:
        conclusion = json.loads(conclusion.strip())
    except Exception as e:
        raise ParseError(f"Invalid conclusion: {conclusion}, error: {e}")
    if not isinstance(conclusion, dict):
        raise ParseError("Conclusion should be a dictionary")
    missing_keys = []
    for key in ['p_true', 'key_factor']:
        if key not in conclusion:
            missing_keys.append(key)
    if len(missing_keys) > 0:
        raise ParseError(f"Missing keys: {', '.join(missing_keys)} in conclusion")
    err = ''
    try:
        p_true = float(conclusion['p_true'])
        assert 0.0 <= p_true <= 1.0, f'p_true should be between 0 and 1'
    except Exception as e:
        err += f'Invalid p_true: {conclusion["p_true"]}, error: {e}\n'
    if not isinstance(conclusion['key_factor'], str):
        err += "key_factor should be a string\n"
    if err != '':
        raise ParseError(err)
    parsed['p_true'] = p_true
    parsed['key_factor'] = conclusion['key_factor']
    parsed['proof'] = parsed['raw'].replace(json_blocks[0], '(OMITTED)').strip()
    return parsed


concluding_prompt = Prompt(
    path='concluding',
    prompt='''Please conclude your analysis into a report of your analysis.

First, provide a comprehensive and informative "Proof" that either proves or disproves the proposition. 
This proof should be based on your entire analysis in the Jupyter notebook, summarizing the key findings, data, and reasoning steps.
**Report Requirements:**
- **Evidence-Based**: You MUST cite specific numbers, dates, and data points from your notebook analysis. Do not just say "data shows increase", say "revenue increased by 15% in Q3 2023".
- **Structure**: Use bullet points to organize key findings, data evidence, and quantitative analysis results.
- **Completeness**: Be as specific and detailed as possible. Do not omit important data or counter-evidence.
- **Visuals**: You are encouraged to present data in markdown tables or reference the plots generated in the notebook.

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
    - **0.90 - 0.99 (Extremely Likely)**: Overwhelming evidence, virtual certainty, negligible risk of error.
    - **0.70 - 0.89 (Very Likely)**: Strong evidence, consistent data, but minor theoretical risks exist.
    - **0.55 - 0.69 (Likely)**: Balance of evidence tilts positive, but significant uncertainties or data gaps remain.
    - **0.50 (Unknown)**: No information, or evidence is perfectly contradictory/balanced.
    - **0.31 - 0.45 (Unlikely)**: Balance of evidence tilts negative; positive drivers are weak or speculative.
    - **0.11 - 0.30 (Very Unlikely)**: Strong counter-evidence, consistent negative data.
    - **0.01 - 0.10 (Extremely Unlikely)**: Overwhelming counter-evidence, virtual impossibility.
    - **0.00 (Impossibility)**: Proven False by direct, irrefutable facts.
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
''',
    md_tags=['json'],
    required_md_tags=['json'],
    parser=concluding_parser,
    allow_web_search=True,
)


