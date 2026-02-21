from openai import OpenAI
import os
from enum import Enum
from analytica.const import Proposition
from analytica.agent.prompts.prover import concluding_parser
from lllm.models import ParseError
import time


OPENAI_API_KEY = os.environ.get("MY_OPENAI_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

class DeepResearchModels(Enum):
    O3_DEEP_RESEARCH = "o3-deep-research"
    O4_MINI_DEEP_RESEARCH = "o4-mini-deep-research-2025-06-26"

MCP_URL = 'https://jserver.ngrok.io/mcp'

DR_SYSTEM_PROMPT = """
You are an expert in analysis.
You will be provided with a proposition, and your task is to provide a comprehensive proof that either proves or disproves the proposition. 
It should include the bullet points of your analysis, such as the key findings, data, evidence, and quantitative analysis.
Please be as specific and detailed as possible, and do not miss any important information, especially the data and evidence.
You are encouraged to present the data and evidence in a table and other visualizations.

Do:
- Focus on data-rich insights: include specific figures, trends, statistics, and measurable outcomes.
- When appropriate, summarize data in a way that could be turned into charts or tables, and call this out in the response.
- Prioritize reliable, up-to-date sources
- Include inline citations and return all source metadata.

Be analytical, avoid generalities, and ensure that each section supports data-backed reasoning.

At the end of your analysis, append a single ```json ... ``` block. Inside this block, provide a single JSON object with exactly two keys:
1.  "p_true": Your estimated probability (a float between 0.00 and 1.00) that the proposition is true, based on your proof and notebook analysis.
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
"""


def _query_dr(proposition: Proposition, model: DeepResearchModels | str) -> str:
    """
    Launch the MCP server in proxy/etc/search_mcp.py
    Then use ngrok to expose the server to the internet.
    Then replace the mcp_url with the ngrok url.
    """
    response = client.responses.create(
        model=model.value if not isinstance(model, str) else model,
        input=[
                {
                "role": "developer",
                "content": [
                    {
                    "type": "input_text",
                    "text": DR_SYSTEM_PROMPT,
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "input_text",
                    "text": f'''Here is the proposition you are going to analyze:
        {proposition.prompt}

        **Data Timeliness**: Remember the current date is {proposition.date}. 

        Now please provide your analysis and proof. Remember to provide the final part of your response in the format of ```json ... ``` block.
        ''',
                    }
                ]
                }
            ],
            reasoning={
                "summary": "auto"
            },
            tools=[
                # {
                #     "type": "web_search_preview",
                # },
                { # ADD MCP TOOL SUPPORT
                    "type": "mcp",
                    "server_label": "search_mcp",
                    "server_url": MCP_URL, # run search mcp server in proxy/etc/search_mcp.py
                    "require_approval": "never"
                },
                {
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",
                        "file_ids": []
                    }
                }
            ]
    )
    return response

def query_dr(proposition: Proposition, model: DeepResearchModels | str, max_retries: int = 3) -> str:
    for _ in range(max_retries):
        try:
            response = _query_dr(proposition, model)
            parsed = concluding_parser(response.output_text)
            return response, parsed
        except ParseError as e:
            print(f"Error querying DR: {e}")
            time.sleep(1)



