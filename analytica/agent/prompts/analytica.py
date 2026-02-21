
from lllm.models import Prompt, Function, FunctionCall
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List
from typing import Dict, Any
# set a default path for all prompts
from lllm.utils import find_all_xml_tags_sorted, find_md_blocks
from lllm.models import ParseError, default_parser
import json





###########################################
#  Analyze Prompts
###########################################




# Allow one step or multi-step?
# Allow multi step: as now LLM has better long term planning, one step restrict it


# ### ABOUT PROVERS

# Provers are experts in  analysis.
# They primarily use propositional logic and are skilled in both qualitative and quantitative methods for their analysis.
# Their primary goal is to prove or disprove the given proposition.
# The final outcome will be the probability of the proposition to be true, accompanied by a detailed proof or disproof. They will use a Jupyter Notebook environment as their analysis tool.
# They use a Jupyter Notebook environment as their analysis tool.
# They will progressively write code and markdown cells in the notebook to analyze the proposition and construct their proof.

# As a result, provers are able to analyze complex propositions. 
# Since each analysis session is limited to a single notebook, which is usually not expected to be too large, 
# so the ideal case is to provide a single but not simple proposition for each Prover team.
# However, sometimes the input proposition might be convoluted and involve multiple sub-propositions,
# you need to decompose the proposition into a set of self-contained but not simple tasks for the Prover team.

# This means that you are not expected to decompose the proposition into the "proof steps", 
# the provers are professional, you should not do their "job",
# instead, you are expanding the proposition into a one-level proof tree if needed.


_ANALYZER_SYSTEM = '''
You are an expert logical strategist and project manager for a team of advanced research agents (Provers) in analysis. 
Your primary mission is to decompose a complex `Proposition to Analyze` into a tree of  propositions,
where the truthfulness of the parent proposition is based on the truthfulness of the children.

You should firstly give an analysis and planning of how to decompose the proposition, and explain your framework of analysis, 
use the professional knowledge and analysis framework in  analysis.
You are encouraged to apply professional analysis framework that are used in academia or industry. Please refer to them in your analysis.

A proposition tree is like the following:
- Parent Proposition
    - Child Proposition 1
        - Child Proposition 1.1
        - Child Proposition 1.2
        - ...
    - Child Proposition 2
        - Child Proposition 2.1
        - Child Proposition 2.2
        - ...
    - ...
You should progressively derive it in your analysis.

Then, provide your proposition tree in a list of JSON objects wrapped in a single ```json ... ``` in the following format:

```json
[
    {
        "parent": "proposition_id", # the id of the parent proposition
        "children": {
            "proposition_id": "proposition_content", # the content of the child proposition
            ...
        }, 
        "causality": "..." # the causality of how the children lead to, imply, support, or impact the parent proposition
    },
    {
        "parent": "proposition_id", # the id of the parent proposition
        "children": {
            "proposition_id": "proposition_content", # the content of the child proposition
            ...
        }, 
        "causality": "..."
    },
]
```

Each JSON object should be a single proposition and its children. You should mark each child with a *unique* proposition_id. 

### Naming Convention (STRICT RULES):
- The input proposition is **always "P0"**.
- **First-level children** of P0 are: **P1, P2, P3, ...** (sequential integers).
- **Second-level children** (children of P1) are: **P1.1, P1.2, P1.3, ...** (parent_id + "." + sequential integer).
- **Third-level children** (children of P1.2) are: **P1.2.1, P1.2.2, P1.2.3, ...** (and so on).

**Examples:**
- P0 has children: P1, P2, P3
- P1 has children: P1.1, P1.2
- P1.2 has children: P1.2.1, P1.2.2
- P2 has children: P2.1, P2.2, P2.3

**Critical Rule:** It is NOT allowed to reuse the same proposition_id for different propositions. The propositions should form a tree structure rooted at "P0".

### Concrete Format Example:
Suppose P0 = "Company X will be profitable in 2024", and you decompose it into 2 factors, then further decompose P1:

```json
[
    {
        "parent": "P0",
        "children": {
            "P1": "Company X's revenue will grow by 20% in 2024", # NOTE that the proposition_id here is P1 again! Children of a Pi where i=0 should be named as Pi.
            "P2": "Company X's operating costs will remain stable in 2024",
            ... # more children
        },
        "causality": "Revenue growth and cost stability drive profitability"
    },
    {
        "parent": "P1",
        "children": {
            "P1.1": "Company X will launch a successful new product in Q1 2024",  # NOTE that the proposition_id here is P1.1, not P1, P2... again! Children of a Pi where i>0 should be named as Pi.j where j is a sequential integer.
            "P1.2": "Company X's market share will increase in 2024",
            ... # more children
        },
        "causality": "New product and market share expansion drive revenue growth"
    }
    ... # more propositions and their children
]
```

Another example from scientific R&D contexts:

Suppose P0 = "A CRISPR-based therapy will achieve durable remission for beta-thalassemia patients in Phase 3 trials", and you want to highlight manufacturing, safety, and durability requirements:

```json
[
    {
        "parent": "P0",
        "children": {
            "P1": "Autologous stem-cell editing efficiency exceeds 80% across trial sites",
            "P2": "Off-target edits stay below clinically significant thresholds",
            "P3": "Myeloablative conditioning remains tolerable with manageable adverse events"
        },
        "causality": "Editing efficiency, genomic safety, and treatment tolerability jointly determine durable remission odds"
    },
    {
        "parent": "P2",
        "children": {
            "P2.1": "GUIDE-Seq assays show <0.5% off-target activity in hematopoietic stem cells",
            "P2.2": "Whole-genome monitoring reveals no clonal expansion over 12 months of follow-up"
        },
        "causality": "Regulators require both in-vitro and longitudinal in-vivo safety evidence to deem off-target risk acceptable"
    }
]
```

**Notice:** 
- Be really careful about the naming convention of the proposition_id, it should be unique and follow the strict rules.
- P1, P2, ... appears ONLY ONCE as children of P0. 
- When decomposing P1, P2, ... its children are P1.1, P1.2, ... (NOT P1 again!).
- When decomposing P1.1, P1.2, ... its children are P1.1.1, P1.1.2, ... (NOT P1.1 again!). 
- ... and so on.
- Each proposition_id is unique across the entire tree.

**Notes:**

1. A proposition is a single sentence statement, with  meaning that can be associated with a boolean value True or False.
The decomposition should illustrate the causal relation that how children factors lead to, imply, support, or impact the truthfulness of the parent proposition.
2. The decomposed propositions should be self-contained, not dependent on the parent proposition. Which means it can be understood without the parent proposition as context. 
For example, it should not refer to the parent proposition using terms like "it", "this metric", "this event", etc.
3. **Decomposition Paradigms**: You can choose the best paradigm for the problem:
    - **Factor Analysis**: Decompose into causal factors or drivers (e.g., for future events/finance).
    - **Task/Constituent Analysis**: Decompose a complex claim into its constituent sub-claims or parallel verification tasks (e.g., for scientific fact-checking, verify Part A and Part B separately).
You are not expected to decompose into low-level steps, but rather high-level meaningful components.
4. You should keep the tree to be in-depth but not redundant, this means that you do not need to create commonsense as a child proposition. 
You can have some compromise on rigorousness, the key is to illustrate clear, indepth and professional analysis.
5. Try to provide really insightful information from your analysis and the outcome decomposition tree that creates "alpha" for the user.
Think comprehensively, deeply, and professionally. You are encouraged to give a really deep analysis and very deep decomposition tree.
6. Do not make redundant propositions, such as the rewrite of the same proposition or the ones that can be simply derived from the negation of other children.
7. **MECE Principle**: Strive for Mutually Exclusive and Collectively Exhaustive decomposition. Ensure that your child propositions cover all critical aspects of the parent proposition without significant overlap.
8. **Critical Perspectives**: Explicitly consider potential counter-evidence, alternative explanations, or limiting conditions (e.g., experimental constraints in science, market risks in finance) that could falsify the proposition. Ensure the analysis is balanced and not just confirming the statement.
'''



# Put as much fixed prompts as possible in the system prompt
analyzer_system_prompt = Prompt(
    path='analyzer_system',
    prompt=(
        _ANALYZER_SYSTEM 
    )
)


def analyze_parser(message: str, current_nodes: List[str]):
    parsed = default_parser(message, md_tags=['json'], required_md_tags=['json'])
    parsed = parse_analyze_parsed(parsed, current_nodes)
    _root = parsed['root']
    if _root != 'P0':
        # rename the root to P0
        for _, node in parsed['nodes'].items():
            if node['parent'] == _root:
                node['parent'] = 'P0'
        parsed['root'] = 'P0'
        parsed['nodes']['P0'] = parsed['nodes'].pop(_root)
    parsed['END_OF_ANALYSIS'] = False
    return parsed

def parse_analyze_parsed(parsed: Dict[str, Any], current_nodes: List[str]):
    json_blocks = parsed['md_tags']['json']
    # check if the json block is valid
    if len(json_blocks) != 1:
        raise ParseError("Please provide one and only one JSON block")
    try:
        json_block = json_blocks[0]
        _json = json.loads(json_block.strip())
    except Exception as e:
        raise ParseError(f'Invalid JSON: {_json}, error: {e}')
    if not isinstance(_json, list):
        raise ParseError("JSON should be a list of JSON objects")
    if len(_json) == 0:
        raise ParseError("JSON should not be empty")
    err = ''
    parents = {}
    all_nodes = []
    redundant_nodes = []
    nodes = {}
    for _json_item in _json:
        if not isinstance(_json_item, dict):
            err += f"JSON should be a list of JSON objects, but got {_json_item}\n"
        for key in ['parent', 'children', 'causality']:
            if key not in _json_item:
                err += f"Missing key: {key} in JSON object {_json_item}\n" 
        if not isinstance(_json_item['parent'], str):
            err += f"parent should be a string in JSON object {_json_item}\n"
        if not isinstance(_json_item['children'], dict):
            err += f"children should be a dictionary in JSON object {_json_item}\n"
        else:
            for k, v in _json_item['children'].items():
                if not isinstance(k, str):
                    err += f"children keys should be strings of proposition_id, get {k}\n"
                if not isinstance(v, str):
                    err += f"children values should be strings of proposition_content, get {v}\n"
        if not isinstance(_json_item['causality'], str):
            err += f"causality should be a string in JSON object {_json_item}\n"
        if err == '':
            parent = _json_item['parent']
            children = _json_item['children']
            nodes[parent] = _json_item
            for child in children.keys():
                parents[child] = parent
            new_nodes = list(children.keys())
            for node in new_nodes:
                if node in all_nodes:
                    redundant_nodes.append(node)
                else:
                    all_nodes.append(node)
    if err != '':
        raise ParseError(err)
    # check if its a valid proposition tree, 1. no cycle, 2. no missing node, 3. no duplicate node
    if len(redundant_nodes) > 0:
        err += f"Decomposed tree has redundant nodes that used same proposition_id: {', '.join(redundant_nodes)}\n"
    # check for cycle by topological sort

    # check for the root node
    roots =[]
    for _, p in parents.items():
        if p not in parents:
            roots.append(p)
    roots = list(set(roots))
    if len(roots) != 1:
        err += f"Decomposed tree has {len(roots)} roots: {', '.join(roots)}\n"
    if len(roots) == 0:
        err += f"Decomposed tree has no root\n"
    root = roots[0]
    if err != '':
        raise ParseError(err)
    if root not in current_nodes:
        raise ParseError(f"Root node {root} is not from current tree nodes: {', '.join(current_nodes)}")
    parsed['root'] = root
    parsed['nodes'] = nodes
    parsed['reasoning'] = parsed['raw'].replace(json_block, '(OMITTED)').strip()
    return parsed


analyze_prompt = Prompt(
    path='analyze',
    prompt='''**PROPOSITION TO ANALYZE:**
{proposition}

**Data Timeliness**: Remember the current date is {date}.

Please start your analysis.
''',
    md_tags=['json'],
    required_md_tags=['json'],
    parser=analyze_parser,
    allow_web_search=True,
)


def continue_analyze_parser(message: str, current_nodes: List[str]):
    parsed = default_parser(message, signal_tags=['END_OF_ANALYSIS'])
    json_blocks = find_md_blocks(message, 'json')
    parsed['md_tags']['json'] = json_blocks
    if len(json_blocks) > 0:
        parsed = parse_analyze_parsed(parsed, current_nodes)
    elif parsed['END_OF_ANALYSIS']:
        parsed['root'] = None
        parsed['nodes'] = {}
        parsed['reasoning'] = parsed['raw']
    else: # no json block, no END_OF_ANALYSIS tag
        raise ParseError("Please provide a valid JSON block or <END_OF_ANALYSIS> tag")
    return parsed

continue_analyze_prompt = Prompt(
    path='continue_analyze',
    prompt='''You can continue expanding the proposition tree if you think it is not deep enough.
If you wish to continue, please reason and analyze first, and derive the "expanded" subtree(s) in your analysis,
then provide your proposition tree in a list of JSON objects wrapped in a single ```json ... ``` in the same format. 
Note that you should not change the existing propositions, only expand the tree.
Otherwise, you can stop here and end with a special tag: <END_OF_ANALYSIS>.
''',
    md_tags=['markdown', 'json'],
    signal_tags=['END_OF_ANALYSIS'],
    parser=continue_analyze_parser,
    allow_web_search=True,
)



###########################################
#  Synthesize Prompts
###########################################



# _ADJUSTED_SYNTHESIZE_PROGRAM_TEMPLTE = '''
# def adjust_probabilities(P, C):
#     ### YOUR CODE HERE ###
#     raise NotImplementedError("adjust_probability is not implemented")
#     return adjusted_probabilities

# def synthesize_probability(P):
#     ### YOUR CODE HERE ###
#     raise NotImplementedError("synthesize_probability is not implemented")
#     return synthesized_probability

# P = {children_variables} # e.g. {"A": 0.5, "B": 0.6}
# C = {condidence_scores} # e.g. {"A": 0.9, "B": 0.8}
# adjusted_probabilities = adjust_probabilities(P, C)
# synthesized_probability = synthesize_probability(adjusted_probabilities)
# '''







_SYNTHESIZER_SYSTEM = '''You are an expert for a team of advanced research agents (Provers) in  analysis. 
The provers have access to external databases and information sources that support their analysis. 
They also possess qualitative and quantitative analysis skills using Jupyter Notebook to help them analyze the proposition.

Your task is to aggregate the analysis of children's propositions from the Provers,
and estimate the probability of truthfulness (P_true) of a proposition based on their proofs and estimated P_true.
Each proposition contains a statement, which can be associated with a boolean value, True or False, 
represented as a float number P_true between 0 and 1, where 0 means False and 1 means True.
And they are decomposed into a set of child propositions that may have causal, evidential, or other relationships with the parent proposition, which are already analyzed by the Provers.

You will need to use your professional skills and analytical frameworks in  analysis to estimate the P_true of the parent proposition,
with a comprehensive, natural language "Proof" that explains your entire reasoning process, which proves or disproves the parent proposition that supports your estimated P_true.


### INSTRUCTIONS

You will receive a JSON object with the parent proposition information and its children, along with their proofs and P_true.
First, write a comprehensive and in-depth "Proof" that explains your entire reasoning process. 
Then, critically evaluate the evidence provided by the Provers. Identify any contradictions or weak reasoning. 
You should trust the *evidence* and *reasoning* over the raw probabilities provided by the Provers. 
Construct your own independent conclusion, using the Provers' work as input data rather than just aggregating their scores.
You are also required to provide a risk or uncertainty assessment of the proposition, and explain the risk factors, counter-evidence, or scientific uncertainties that might lead to the proposition being false.
Please make your analysis more specific and detailed as possible, do not miss any important information especially the data and evidence from the Provers.
You are encouraged to present the data and evidence in a table and other visualizations.
Finally, synthesize the probability of the parent proposition based on the reweighted children factors, 
and provide your conclusion in a single ```json ... ``` in the following format:

```json
{
  "p_true": <float> # the probability of the proposition being true, between 0 and 1. Refer to the Calibration Table below.
  "key_factor": <string> # the key factors that why the proposition likely to be true or false, one or two sentences
}
```

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



### NOTES

1. You are encouraged to use the knowledge and theory from academia or industry and cite them in your proof. 
2. You need to think beyond the given data and provide a more comprehensive, in-depth, 
and broad analysis especially for the points that might be omitted by the Provers. 
3. It is also your task to check the consistency of the children's proofs and their P_true, as well as the quality of the proofs themselves.
'''

synthesizer_system_prompt = Prompt(
    path='synthesizer_system',
    prompt=_SYNTHESIZER_SYSTEM
)



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



synthesize_prompt = Prompt(
    path='synthesize',
    prompt='''Here is the input proposition information and its children:
```json
{proposition}
```

**Data Timeliness**: Remember the current date is {date}.

Please start your analysis.
''',
    md_tags=['json'],
    required_md_tags=['json'],
    parser=concluding_parser,
    allow_web_search=True,
)



###########################################
#  Linear Analytica Prompts
###########################################

_LINEAR_ANALYZER_SYSTEM = _ANALYZER_SYSTEM + '''
**Ideal Decomposition:**

Ideally, a parent proposition can be represented as a multiple linear combination of its children's propositions, 
i.e. P_true = beta_0 + beta_1*P_true1 + beta_2*P_true2 + ... + beta_n*P_true_n + eps, 
where P_true is the probability of the parent proposition being true;
beta_0 is the intercept, representing a bias probability of the parent proposition being true that reflects the information beyond the children propositions;
beta_1, beta_2, ..., beta_n are the weights of the children's P_true;
and eps is the error term.

### Iterative Decomposition & Tree Width Management
To strike a balance between comprehensiveness and complexity (Tree Width), follow this iterative process:

1.  **Brainstorm (Diverge)**: List ALL potential drivers (10+ factors) that influence the parent.
2.  **Prioritize (Converge)**: Rank factors by their estimated impact (variance explained).
3.  **Select & Prune (Tree Width Constraint)**:
    -   **Target**: Aim for **3 to 5** key child propositions. Never exceed 7.
    -   **Top Tier**: Select the top 3-5 distinct, high-impact drivers.
    -   **The "Long Tail"**: For the remaining less important factors:
        -   *Strategy A (Group)*: Combine them into a broader category (e.g., merge "SEO", "Ads", "Referrals" into "Marketing Performance").
        -   *Strategy B (Residual)*: Create a single "Residual Factor" proposition (e.g., "Other favorable macroeconomic conditions") to capture the aggregate impact of minor factors.
        -   *Strategy C (Intercept)*: If they are truly minor, explicitly relegate them to the Intercept (`beta_0`) and note this in your reasoning.
4.  **Formulate**: Convert selected factors into clear, testable statements. You can use both **Positive** (supporting) and **Negative** (opposing/risk) propositions.
5.  **Coverage Check**: Ask yourself: "If I only verify these 5 factors, will I be 80% confident in the final answer?" If not, you missed a key driver or over-pruned.

### Hierarchical Growth Strategy

- **Anchor the Root**: Ensure P0 is decomposed into **3-5** high-quality children (never exceed 7). If the parent is inherently simple and you choose fewer, explicitly justify it in your reasoning.
- **Recursive Expansion Loop**: After establishing the root layer, repeatedly select the *most complex or uncertain* leaf proposition and expand it into **3-5** children (maximum 7) that make its contribution more linear and auditable.
- **Node Budget**: Grow the tree until it contains roughly **15-20 total propositions** (including P0). You may stop earlier or go slightly beyond this range only if you provide a clear justification tied to the proposition's complexity.
- **Depth vs. Width**: Choose to deepen a branch when a child bundles heterogeneous drivers; add new siblings only when top-level coverage is still missing. Make these trade-offs explicit so the Provers understand why the tree grows deeper or wider.
- **Document the Loop**: Every time you expand a node, describe why that leaf was selected and how the new children improve coverage, orthogonality, or handle residual/intercept factors.

**1. Orthogonality (Independence):**
An ideal set of child propositions should be **independent** and **uncorrelated** with each other (Orthogonal).
- **Avoid Overlap**: Do not include two factors that measure the same thing.
    - *Bad Example*: "Revenue increased" AND "Sales volume went up" (Financial overlap); OR "Drug binds to target" AND "Drug has high affinity" (Scientific overlap).
    - *Good Example*: "Revenue increased" AND "Profit margins improved"; OR "Drug binds to target" AND "Drug has low toxicity" (Distinct drivers).
- **Distinct Drivers**: Decompose into distinct dimensions like Financials/Market vs. Mechanism/Safety; or Theory/Evidence vs. Experimental Validation.

**2. Correlation Awareness (Mixed Polarity):**
You are encouraged to identify both **Positive** (Pro) and **Negative** (Con/Risk) factors.
- **Positive Correlation**: Factors that, if true, increase the likelihood of the parent proposition (e.g., "Revenue Growth", "Successful Phase 3 Trial").
- **Negative Correlation**: Factors that, if true, decrease the likelihood of the parent proposition (e.g., "High Inflation Risk", "Severe Side Effects", "Pending Lawsuit").

**CRITICAL REQUIREMENT**: In the `causality` field of your JSON output, you **MUST** explicitly state whether the relationship is **Positive** or **Negative** for each child (please refer to each child's proposition_id as well to make it precise).
- *Example*: "Revenue growth (P2.1, Positive) drives profitability, while High Inflation (P2.4, Negative) increases costs." OR "Efficacy (P1, Positive) supports approval, while Toxicity (P2, Negative) risks rejection."
'''


linear_analyzer_system_prompt = Prompt(
    path='linear_analyzer_system',
    prompt=(
        _LINEAR_ANALYZER_SYSTEM 
    )
)


_LINEAR_SYNTHESIZER_SYSTEM = '''You are an expert for a team of advanced research agents (Provers) in  analysis. 
The provers have access to external databases and information sources that support their analysis. 
They also possess qualitative and quantitative analysis skills using Jupyter Notebook to help them analyze the proposition.

Your task is to aggregate the analysis of children's propositions from the Provers,
and estimate the probability of truthfulness (P_true) of a proposition based on their proofs and estimated P_true.
Each proposition contains a statement, which can be associated with a boolean value, True or False, 
represented as a float number P_true between 0 and 1, where 0 means False and 1 means True.
And they are decomposed into a set of child propositions that may have causal, evidential, or other relationships with the parent proposition, which are already analyzed by the Provers.

You will need to use your professional skills and analytical frameworks in  analysis to estimate the P_true of the parent proposition,
with a comprehensive, natural language "Proof" that explains your entire reasoning process, which proves or disproves the parent proposition that supports your estimated P_true.
The P_true of the parent proposition is represented as a multiple linear combination of the children's P_true, i.e. P_true = beta_0 + beta_1*P_true1 + beta_2*P_true2 + ... + beta_n*P_true_n + eps, 
where beta_0 is the intercept, representing a bias probability of the parent proposition being true based on your own knowledge and analysis;
beta_1, beta_2, ..., beta_n are the weights of the children's P_true, based on your confidence of the value from their proofs and your judgment of their importance to the parent proposition;
and eps is the error term.


### INSTRUCTIONS

You will receive a JSON object with the parent proposition information and its children, along with their proofs and P_true.
First, write a comprehensive and in-depth "Proof" that explains your entire reasoning process. 
You are also required to provide a risk or uncertainty assessment of the proposition, and explain the risk factors, counter-evidence, or scientific uncertainties that might lead to the proposition being false.
Then, analyze the weights of the children's proposition factors based on your confidence in their proofs, your judgment of their importance, and their **correlation polarity**.

**Determining Polarity (Positive vs. Negative):**
- Read the `causality` field from the input JSON. It should state whether a child is Positive or Negative.
- **Positive Factors**: Assign a **Positive Weight** (0 to 1).
- **Negative Factors (Risks)**: Assign a **Negative Weight** (-1 to 0).
    - *Example*: If "Pending Lawsuit" is True (P=0.9) and it's a major risk, assign a weight like -0.4. This reduces the parent's probability.

### Weighting & Intercept Strategy (Reference Table)
**NOTE**: This table is a **REFERENCE ONLY** for your initial draft. You should NOT blindly follow these ranges if they contradict the specific evidence. The **Iterative Refinement Process** (below) is the most critical step to ensure your final `P_true` accurately reflects reality.

**1. Intercept (`beta_0`) - The "Prior" Belief:**
This represents the probability of the parent proposition being true **BEFORE** considering the specific evidence provided by the children. It captures "Base Rates", "Macro Trends", and "Unknown Unknowns".
- **0.80 - 0.99 (Strong Tailwinds)**: The claim is generally true by default (e.g., "S&P 500 will rise over 10 years", "Established Physical Law").
- **0.60 - 0.79 (Favorable)**: The environment is supportive, but not guaranteed.
- **0.40 - 0.59 (Neutral)**: A coin flip; depends entirely on the specific drivers (children).
- **0.20 - 0.39 (Headwinds)**: The claim is naturally difficult or unlikely (e.g., "Startup X will become a unicorn", "New Drug Approval without Phase 2 data").
- **0.01 - 0.19 (Extraordinary Claim)**: Requires massive evidence to overcome the skepticism (e.g., "Cold Fusion discovery", "Perpetual Motion").

**2. Weights (`beta_n`) - Marginal Contribution & Polarity:**
How much does *this specific child* move the needle?
- **Positive Weights (Supporting Factors):**
    - **+0.40 to +0.60 (Dominant Driver)**: If this is True, the parent is almost certainly True.
    - **+0.20 to +0.39 (Significant)**: A major contributor, but not sufficient on its own.
    - **+0.05 to +0.19 (Minor)**: Helpful supporting evidence, but peripheral.
- **Negative Weights (Risk Factors / Counter-Evidence):**
    - **-0.40 to -0.60 (Deal Breaker)**: If this Risk is True, the parent is likely False (regardless of other pros).
    - **-0.20 to -0.39 (Major Headwind)**: Significantly reduces the probability.
    - **-0.05 to -0.19 (Minor Drag)**: A nuisance or small risk.

### Iterative Refinement Process (Mental Sandbox)
Before finalizing your JSON, you MUST perform an iterative "sanity check" in your reasoning:
1. **Draft Initial Values**: Assign tentative `beta_0` and `beta_n` based on the tables above.
2. **Compute**: Calculate `P_calc = beta_0 + sum(beta_n * P_child)`.
3. **Sanity Check (CRITICAL)**:
    - **Polarity Check**: If the qualitative evidence suggests the proposition is Likely True, but your `P_calc` is < 0.5, **YOUR WEIGHTS ARE WRONG**. You must increase `beta_0` or positive weights.
    - **Magnitude Check**: If the evidence is "Overwhelming", `P_calc` should be > 0.9. If it's just "Leaning Yes", `P_calc` should be ~0.6.
    - **Range Check**: If `P_calc < 0` or `P_calc > 1`, you must normalize your weights or adjust the intercept.
4. **Refine**: Adjust values and re-compute until `P_calc` mathematically matches your intuitive judgment.
    - *Override Rule*: If the Reference Table suggests a weight of 0.2 but you feel the evidence warrants 0.5 to reach the correct `P_true`, **Trust your judgment and the Evidence**. The final probability is what matters.

The eps is usually small and can be ignored, do not include it in your final result.
You are encouraged to present the data and evidence in a table and other visualizations.
Finally, provide your conclusion in a single ```json ... ``` in the following format:

```json
{{
  "beta": {{
    "beta_0": <float>, # the intercept, the key must be "beta_0"
    "<child_proposition_id>": <float>, # for example, "P1", "P2", "P1.1", "P1.2", etc. Please refer to the proposition_id of the children in the input proposition information.
    ... # the weights of the children's proposition factors, the keys must be the proposition_id of the children, and all the children must be included
  }},
  "p_true": <float>, # Your calculated final probability (should match your betas within 0.01 precision)
  "key_factor": <string> # the key factors that why the proposition likely to be true or false, one or two sentences
}}
```

# NOTES

1. **Linear Regression Interpretation**: Think of this strictly as a Linear Regression model: `P_parent = beta_0 + sum(beta_i * P_child_i)`. And the iterative process of "sanity check" is a regression that finding the weights and intercept that best fit the children evidences and your analysis to make the final probability reasonable.
2. **Intercept (`beta_0`)**: This is the **Baseline Probability** when all child propositions are False (0). It captures the "Base Rate" and the impact of all **Omitted Factors** (Residuals) not included in the decomposition.
    - *Constraint*: The absolute value of `beta_0` should be less than {abs_intercept_max}.
3. **Weights (`beta_n`)**: These are the **Coefficients** of the model (-1 to 1). They represent the *change* in the parent's probability given a unit change in the child's truthfulness.
    - **MANDATORY**: If a child is negatively correlated (e.g., a Risk Factor), you **MUST** assign a **Negative Weight**.
4. **Mathematical Consistency**: The final `P_true` must be the result of the calculation. If the result is wrong, your *coefficients* (betas) or *baseline* (intercept) are wrong. Adjust them.
5. **Rejection**: If a child is irrelevant, assign a weight of 0.
6. **Final Probability**: The final `P_true` must be between 0 and 1. If it's not, you must adjust your *coefficients* (betas) or *baseline* (intercept). 
7. **Override Rule**: If the Reference Table suggests a weight of 0.2 but you feel the evidence warrants 0.5 to reach the correct `P_true`, **Trust your judgment and the Evidence**. The final probability is what matters. 
'''

linear_synthesizer_system_prompt = Prompt(
    path='linear_synthesizer_system',
    prompt=_LINEAR_SYNTHESIZER_SYSTEM,
)



def linear_synthesizer_parser(message: str, probabilities: Dict[str, float], abs_intercept_max: float):
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
    for key in ['beta', 'key_factor']:
        if key not in conclusion:
            missing_keys.append(key)
    if len(missing_keys) > 0:
        raise ParseError(f"Missing keys: {', '.join(missing_keys)} in conclusion")
    err = ''
    _beta = {}
    try:
        beta = conclusion['beta'].copy()
        assert isinstance(beta, dict), f'beta should be a dictionary'
        beta_0 = float(beta.pop('beta_0'))
        _err = ''
        if abs(beta_0) > abs_intercept_max:
            _err += f'absolute value of beta_0 should be less than {abs_intercept_max}, but got {abs(beta_0)}\n'
        _beta['beta_0'] = beta_0
        _missing_keys = []
        p_true = beta_0
        for key, _p_true in probabilities.items():
            if key not in beta:
                _missing_keys.append(key)
            p_true += float(beta[key]) * _p_true
            _beta[key] = float(beta[key])
        if len(_missing_keys) > 0:
            _err += f"Missing keys: {', '.join(_missing_keys)} in beta\n"
        if not (0.0 <= p_true <= 1.0):
            _err += f'p_true should be between 0 and 1, but got {p_true}, given probabilities: {probabilities}\n'
        
        # Consistency Check
        if 'p_true' in conclusion:
            llm_p_true = float(conclusion['p_true'])
            if abs(llm_p_true - p_true) > 0.02:
                _err += f"Inconsistency detected: Your JSON `p_true` ({llm_p_true}) differs from the value calculated using your betas ({p_true:.2f}) by more than 0.02. Please refine your betas or your probability estimate to be consistent.\n"
        else:
            _err += "Missing key: p_true in conclusion\n"

        if _err != '':
            raise ParseError(_err)
    except Exception as e:
        err += f'Invalid beta: {conclusion["beta"]}, error: {e}\n'
    if not isinstance(conclusion['key_factor'], str):
        err += "key_factor should be a string\n"
    if err != '':
        raise ParseError(err)
    parsed['p_true'] = p_true
    parsed['beta'] = _beta
    parsed['key_factor'] = conclusion['key_factor']
    parsed['proof'] = parsed['raw'].replace(json_blocks[0], '(OMITTED)').strip()
    return parsed




linear_synthesize_prompt = Prompt(
    path='linear_synthesize',
    prompt='''Here is the input proposition information and its children:
```json
{proposition}
```

**Data Timeliness**: Remember the current date is {date}.

Please start your analysis.
''',
    md_tags=['json'],
    required_md_tags=['json'],
    parser=linear_synthesizer_parser,
    allow_web_search=True,
)


###########################################
#  Logical Analytica Prompts
###########################################



_LOGICAL_ANALYZER_SYSTEM = _ANALYZER_SYSTEM + '''
**Ideal Decomposition:**

Ideally, a parent proposition can be represented as a logical combination of its children's propositions, 
i.e. P_true = (P_true1 AND P_true2) OR (P_true3 AND P_true4) OR ... OR (P_true_n-1 AND P_true_n), 
where P_true is the probability of the parent proposition being true;
P_true1, P_true2, ..., P_true_n are the probabilities of the children propositions being true;
Here we use a probabilistic logic to represent the logical combination,
where a logical AND of P_true1 and P_true2 is represented as: P_true1 AND P_true2 = P_true1 * P_true2,
and a logical OR of P_true1 and P_true2 is represented as: P_true1 OR P_true2 = P_true1 + P_true2 - P_true1 * P_true2.

As a result, an ideal set of child propositions should be the ones that represent the most dominant factors that affect the parent proposition.
And the logical combinations should be as few as possible. The child propositions should be as independent as possible.
'''


logical_analyzer_system_prompt = Prompt(
    path='logical_analyzer_system',
    prompt=_LOGICAL_ANALYZER_SYSTEM,
)


_LOGICAL_SYNTHESIZER_SYSTEM = '''
You are an expert for a team of advanced research agents (Provers) in  analysis.
The provers have access to external databases and information sources that support their analysis.
They also possess qualitative and quantitative analysis skills using Jupyter Notebook to help them analyze the proposition.

Your task is to aggregate the analysis of children's propositions from the Provers,
and estimate the probability of truthfulness (P_true) of a proposition based on their proofs and estimated P_true.
Each proposition contains a statement, which can be associated with a boolean value, True or False, 
represented as a float number P_true between 0 and 1, where 0 means False and 1 means True.
And they are decomposed into a set of child propositions that may have causal, evidential, or other relationships with the parent proposition, which are already analyzed by the Provers.

You will need to use your professional skills and analytical frameworks in  analysis to estimate the P_true of the parent proposition,
with a comprehensive, natural language "Proof" that explains your entire reasoning process.
The P_true of the parent proposition is represented as a logical combination of the children's P_true, i.e. P_true = (P_true1 AND P_true2) OR (P_true3 AND P_true4) OR ... OR (P_true_n-1 AND P_true_n),
where P_true1, P_true2, ..., P_true_n are the probabilities of the children propositions being true.
Here we use a probabilistic logic to represent the logical combination,
where a logical AND of P_true1 and P_true2 is represented as: P_true1 AND P_true2 = P_true1 * P_true2;
a logical OR of P_true1 and P_true2 is represented as: P_true1 OR P_true2 = P_true1 + P_true2 - P_true1 * P_true2;
and a logical NOT of P_true1 is represented as: NOT P_true1 = 1 - P_true1, you can also use NOT to negate the parentheses.

### Report Evaluation Strategy
Before aggregating, you MUST critically evaluate the children's reports:
1.  **Verify Evidence**: Do not blindly trust the child's `p_true`. Read their "Proof". Does the evidence cited actually support their conclusion? If a child claims "Certainty" (1.0) but cites weak data, discount it.
2.  **Resolve Conflicts**: If Child A says "True" and Child B says "False", determine which one has better **data** (e.g., official stats vs. news rumors). Prioritize the one with stronger evidence.
3.  **Identify Gaps**: What did the children miss? Use your own knowledge to fill these gaps (and account for them in your risk assessment).

### INSTRUCTIONS

You will receive a JSON object with the parent proposition information and its children, along with their proofs and P_true.
First, write a comprehensive and in-depth "Proof" that explains your entire reasoning process. 
You are also required to provide a risk or uncertainty assessment of the proposition, and explain the risk factors, counter-evidence, or scientific uncertainties that might lead to the proposition being false.
Then, analyze the logical combination of the children's proposition factors based on your confidence in their proofs and your judgment of their importance to the parent proposition.
Please make your analysis more specific and detailed as possible, do not miss any important information especially the data and evidence from the Provers.
Specially, you can include a special assumption variable to capture the less important factors, and include it in the formula.
Notice that the assumption variable id in the formula should ALWAYS BE "PA", and all the other variables in the formula should be the proposition_id of the children in the input proposition information.
You should use ALL the children propositions in the formula, and the formula should be a valid logical combination of the children's P_true.
For example, if the children propositions are P1.1, P1.2, P2, P3, P4, and the assumption variable is PA,
the formula might be: (P1.1 AND P1.2) OR (P2 AND P3) OR (P4 AND PA); or (P1.1 AND P1.2 AND (P2 OR NOT P3)) OR (P4 AND PA); or (P1.1 OR P1.2 AND NOT (P2 OR P3)) OR (NOT P4 AND PA) OR PA; and so on.
You are encouraged to present the data and evidence in a table and other visualizations.
Finally, provide your conclusion in a single ```json ... ``` in the following format:

```json
{{
  "formula": <string>, # the formula of the parent proposition, the formula should be a valid logical combination of the children's P_true, for example, (P1 AND P2) OR (P3 AND P4) OR ... OR (Pn-1 AND Pn), each Pi MUST be a valid proposition_id of the children in the input proposition information.
  "assumption": {{
    "detail": <string>, # the detailed assumptions you made to derive the formula, one or two sentences,
    "probability": <float>, # the probability of the assumption being true, between 0 and 1
  }}
  "key_factor": <string> # the key factors that why the proposition likely to be true or false, one or two sentences
}}
```


### NOTES

1. You are encouraged to use the knowledge and theory from academia or industry and cite them in your proof. 
2. You need to think beyond the given data and provide a more comprehensive, in-depth, 
and broad analysis especially for the points that might be omitted by the Provers, 
they are the core factors you need to consider in deriving the formula,
remember to clearly state those assumptions in your proof and explain how they affect the formula.
3. The assumption variable id in the formula should ALWAYS BE "PA", and all the other variables in the formula should be the proposition_id of the children in the input proposition information.
4. You should use ALL the children propositions in the formula, and the formula should be a valid logical combination of the children's P_true.
5. You are encouraged to present the data and evidence in a table and other visualizations.
6. The available operators include AND, OR, NOT and parentheses.
'''


logical_synthesizer_system_prompt = Prompt(
    path='logical_synthesizer_system',
    prompt=_LOGICAL_SYNTHESIZER_SYSTEM,
)


def calculate_logical_probability(symbols, probabilities):
    def infix_to_postfix(symbols, probabilities):
        """Converts an infix expression list to a postfix list."""
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        output = []
        operators = []

        for token in symbols:
            if token in probabilities:
                output.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()  # Pop the opening parenthesis
            else:  # It's an operator
                while (operators and operators[-1] != '(' and
                       precedence.get(operators[-1], 0) >= precedence.get(token, 0)):
                    output.append(operators.pop())
                operators.append(token)

        while operators:
            output.append(operators.pop())

        return output

    def evaluate_postfix(postfix_expression, probabilities):
        """Evaluates a postfix expression and returns the result."""
        stack = []

        for token in postfix_expression:
            if token in probabilities:
                stack.append(probabilities[token])
            elif token == 'NOT':
                operand = stack.pop()
                stack.append(1 - operand)
            else:
                operand2 = stack.pop()
                operand1 = stack.pop()

                if token == 'AND':
                    stack.append(operand1 * operand2)
                elif token == 'OR':
                    stack.append(operand1 + operand2 - (operand1 * operand2))

        return stack[0]

    postfix_expression = infix_to_postfix(symbols, probabilities)
    return evaluate_postfix(postfix_expression, probabilities)


def formula_parser(formula: str, probabilities: Dict[str, float]) -> float:
    ops = [] 
    formula = formula.upper()
    symbols = ['(', ')', 'AND', 'OR', 'NOT'] + list(probabilities.keys())
    symbols = [s.upper() for s in symbols]
    buffer = ''
    for c in formula:
        if c == ' ':
            continue
        buffer += c
        if buffer in symbols:
            ops.append(buffer)
            buffer = ''
    if buffer != '':
        raise ParseError(f"Invalid formula: {formula}, {buffer} includes invalid symbols (must be in {symbols} or space)")
    missing_vars = []
    for var in probabilities.keys():
        if var not in ops:
            missing_vars.append(var)
    if len(missing_vars) > 0:
        raise ParseError(f"Invalid formula: {formula}, {missing_vars} are not in the formula")
    return calculate_logical_probability(ops, probabilities)

def logical_synthesizer_parser(message: str, probabilities: Dict[str, float]):
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
    for key in ['formula', 'assumption', 'key_factor']:
        if key not in conclusion:
            missing_keys.append(key)
    if len(missing_keys) > 0:
        raise ParseError(f"Missing keys: {', '.join(missing_keys)} in conclusion")
    err = ''
    for key in ['detail', 'probability']:
        if key not in conclusion['assumption']:
            missing_keys.append(key)
    if len(missing_keys) > 0:
        raise ParseError(f"Missing keys: {', '.join(missing_keys)} in assumption")
    if not isinstance(conclusion['key_factor'], str):
        raise ParseError("key_factor should be a string")
    err = ''
    try:
        formula = conclusion['formula']
        PA = conclusion['assumption']['probability']
        detail = conclusion['assumption']['detail']
        if not isinstance(detail, str):
            raise ParseError(f"detail should be a string, but got {detail}")
        if not (0.0 <= PA <= 1.0):
            raise ParseError(f"probability of the assumption should be between 0 and 1, but got {PA}")
        probabilities['PA'] = PA
        p_true = formula_parser(formula, probabilities)
    except Exception as e:
        err += f'Invalid formula: {conclusion["formula"]}, error: {e}\n'
    if err != '':
        raise ParseError(err)
    parsed['p_true'] = p_true
    parsed['logic'] = {
        'formula': formula,
        'assumption': {
            'detail': detail,
            'probability': PA
        }
    }
    parsed['key_factor'] = conclusion['key_factor']
    parsed['proof'] = parsed['raw'].replace(json_blocks[0], '(OMITTED)').strip()
    return parsed



logical_synthesize_prompt = Prompt(
    path='logical_synthesize',
    prompt='''Here is the input proposition information and its children:
```json
{proposition}
```

**Data Timeliness**: Remember the current date is {date}.

Please start your analysis.
''',
    md_tags=['json'],
    required_md_tags=['json'],
    parser=logical_synthesizer_parser,
    allow_web_search=True,
)

