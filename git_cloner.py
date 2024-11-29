#!/bin/
# 1. Copy this file into the Colab machine right after starting a new collab session
# 2. Then run it via terminal, with a command such as:
#    ipython git_cloner.py
# 3. This will clone the repo in the Colab machine

import os
import subprocess
from subprocess import Popen

from google.colab import userdata

REPO_URL = "https://github.com/maryialejandra/llm_finetunning"
REPO_LOCAL_DIR = REPO_URL.split("/")[-1]

def on_import():

    print('Creating .git-credentials file using Colab secret GITHUB_USERPWD')
    with open('/root/.git-credentials', 'wt') as f_out:
        print(userdata.get('GITHUB_USERPWD'), file=f_out)

    # !ls -lrt /root/.git-credentials
    print("Contents of /roo/.git-credentials:")
    # subprocess.call(['ls', '-lrt', '/root/.git-credentials'])
    run_cmd('ls -lrt /root/.git-credentials')

    # !git config --global credential.helper store':
    # subprocess.call(['git', 'config', '--global', 'credential.helper', 'store'])
    run_cmd('git config --global credential.helper store')
    # %cd /content
    os.chdir('/content')

    repo_local_dir = REPO_LOCAL_DIR
    if not os.path.exists(repo_local_dir):
        print(f"{repo_local_dir} directory does not exist. Attempting git clone...")
        clone()
    else:
        print(f"{repo_local_dir} directory already exists. Attempting git pull.." )
        pull()

    print(f"Current working directory is: {os.getcwd()}")


def clone():
    repo_local_dir = REPO_LOCAL_DIR
    os.chdir('/content')
    run_cmd(f"git clone {REPO_URL} ./{repo_local_dir}")
    os.chdir(repo_local_dir)
    print(f"\nRepo local dir ({repo_local_dir}) contents:")
    run_cmd('ls -lrt ./')

def pull():
    repo_local_dir = REPO_LOCAL_DIR
    os.chdir(repo_local_dir)
    subprocess.call(['git', 'pull'])


def run_cmd(a_str: str):
    cmd_parts = a_str.split(" ")
    p = Popen(cmd_parts, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

    stdout, stderr = p.communicate()
    if stdout and len(stdout) > 0:
        print("STDOUT:")
        for line in stdout.split(b"\n"):
            print(try_decode(line))

    if stderr and len(stderr) > 0:
        print("stderr:", stderr)
        for line in stderr.split(b"\n"):
            print(try_decode(line))


def try_decode(line: bytes) -> str:
    try:
        return line.decode("ascii")
    except Exception:
        try:
          return line.decode("utf8")
        except Exception:
          return f"FAILED decoding: {line!r}"




on_import()
