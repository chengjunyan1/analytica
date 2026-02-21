cloc --include-ext=py \
bin/ \
cli.py \
analytica/utils.py \
analytica/system.py \
analytica/const.py \
analytica/agent/ae.py \
analytica/agent/prompts/ \
analytica/proxy/re.py \
analytica/proxy/tools.py \
lllm/ \





# Count only the core code (the code that will be used within the evolution)
# exclude similar code (e.g. claude.py) and unused code (e.g. reviewer.py)
# may include little non-core code (e.g. some parts from tester.py)
# there is actually some code from customized lm-eval, not counted here
# some codes are adapted elsewhere, they are not counted, e.g. trainer.py, etc., only few are counted, e.g. modules.py