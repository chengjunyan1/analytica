#!/usr/bin/env python3
import click
import subprocess
import sys

# call it by: analytica xxx

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    """Analytica CLI tool for running configuration, node, and evolution commands."""
    pass

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), add_help_option=False)
@click.pass_context
def gui(ctx):
    """Run the Streamlit GUI"""
    cmd = ['streamlit', 'run', 'bin/app.py'] + ctx.args
    subprocess.run(cmd)

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), add_help_option=False)
@click.pass_context
def setup(ctx):
    """Setup the environment""" # bash scripts/setup_env.sh
    cmd = ['bash', 'scripts/setup_env.sh']
    subprocess.run(cmd)

@cli.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), 
    add_help_option=True,
    help='''Test the runtime environment, usage: analytica testre --cutoff_date, -c [optional, e.g. 2025-01-01] --test, -t [optional, default=all]'''
)
@click.option('--test_proxy', '-tp', default=None, help='Test type or path (default: None), all means test all proxies')
@click.option('--cutoff_date', '-c', default=None, help='Cutoff date, e.g., 2025-01-01')
@click.option('--skip_k', '-s', default='0', help='Skip k endpoints (default: 0)')
@click.option('--test_maker', '-tm', is_flag=True, help='Test the maker (default: False)')
def testre(test_proxy, cutoff_date, skip_k, test_maker):
    """Test the runtime environment""" # analytica testre --cutoff_date 2024-01-01 --test fmp --skip_k 35
    cmd = ['python', 'analytica/proxy/re.py']
    if cutoff_date is not None:
        cmd += ['--cutoff_date', cutoff_date]
    if test_proxy is not None:
        cmd += ['--test_proxy', test_proxy]
    if test_maker:
        cmd.append('--test_maker')  # no value needed
    if skip_k is not None:
        cmd += ['--skip_k', skip_k]
    subprocess.run(cmd)
    

# TODO: set default

if __name__ == '__main__':
    cli()

