#!/bin/
# 1. Copy this file into the Colab machine right after starting a new collab session
# 2. Then run it via terminal, with a command such as:
#    ipython git_cloner.py
# 3. This will clone the repo in the Colab machine

import os
from pathlib import Path
import subprocess

from google.colab import userdata

REPO_URL = "https://github.com/maryialejandra/llm_finetunning"

def on_import():

    print('Creating .git-credentials file using Colab secret GITHUB_USERPWD')
    with open('/root/.git-credentials', 'wt') as f_out:
        print(userdata.get('GITHUB_USERPWD'), file=f_out)

    # !ls -lrt /root/.git-credentials
    print("Contents of /roo/.git-credentials:")
    subprocess.call(['ls', '-lrt', '/root/.git-credentials'])

    # !git config --global credential.helper store':
    subprocess.call(['git', 'config', '--global', 'credential.helper', 'store'])

    # %cd /content
    os.chdir('/content')

    repo_local_dir = REPO_URL.split("/")[-1]

    if not os.path.exists(repo_local_dir):
        print(f"{repo_local_dir} directory does not exist. Attempting git clone")
        subprocess.call(["git" "clone", REPO_URL, "./llm_finetunning"])
        os.chdir(repo_local_dir)
        subprocess.call(['ls', '-lrt', './'])
    else:
        print( f"{repo_local_dir} directory already exists. Attempting git pull" )
        os.chdir(repo_local_dir)
        subprocess.call(['git', 'pull'])


def run_cmd(a_str: str):
    cmd_parts = a_str.split(" ")
    subprocess.call(cmd_parts, stdout=subprocess.PIPE)


on_import()
