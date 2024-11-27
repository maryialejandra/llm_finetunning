

The following are instructions on how to git clone our private repo from within google colab.

0. *Prerequisite:* Make sure you can do git clone of our repo in your own computer first and that you have set up git credential store. This last bit is crucial for the next step

    ```bash
    # on local computer do:

    git config --global credential.helper store  # You _need_ this for next step!

    # cd to directory where you want to clone the repo
    git clone https://github.com/maryialejandra/llm_finetunning ./llm_finetunning

    # type in your github username and password (or token) if/when prompted
    ```
1. Assuming previous step worked, your github credentials should now be stored in a file called  `.git-credentials` directly under your local home directory.

   ```bash
   $ cat $HOME/.git-credentials  # or just open it from a text editor

   https://<YOUR-USERNAME-HERE>:<YOUR-PASSWORD-OR-TOKEN-HERE>@github.com
   ```

2. If you have ever cloned repos from other domains different from `github.com`, `.git-credentials`  might contain some other lines.
   However, _the only line we care about is the line that ends in `github.com` and that contains the username and password you used in step 0._

   Now go to google colab and create a secret (using the little key icon that you will see on the left margin
   - if you don't see it do: Ctrl + Shift + P to bring up the command palette and then type "Open user secrets" and click on the command).
   The secret name should be `GITHUB_USERPWD` and as its value paste the _whole_ line from `.git-credentials` we just mentioned.
   Make sure to strip any spaces at the beginning or end when pasting it.

3. Now you can clone our repo from google colab running the following code in a cell:

    ```python
    from google.colab import userdata  # get access to secrets defined in secrets tab
    with open('/root/.git-credentials', 'wt') as f_out:
        print(userdata.get('GITHUB_USERPWD'), file=f_out)

    !ls -lrt /root/.git-credentials
    !git config --global credential.helper store
    %cd /content
    !rm -rf llm_finetunning
    !git clone  https://github.com/maryialejandra/llm_finetunning ./llm_finetunning
    !ls -lrt llm_finetunning
    ```

    As a result you should see an output like this:
    ```bash
    -rw------- 1 root root 73 Nov 27 10:07 /root/.git-credentials
    /content
    Cloning into './llm_finetunning'...
    remote: Enumerating objects: 99, done.
    remote: Counting objects: 100% (99/99), done.
    remote: Compressing objects: 100% (69/69), done.
    remote: Total 99 (delta 43), reused 80 (delta 26), pack-reused 0 (from 0)
    Receiving objects: 100% (99/99), 162.03 KiB | 14.73 MiB/s, done.
    Resolving deltas: 100% (43/43), done.
    total 20
    -rw-r--r-- 1 root root   17 Nov 27 10:07 README.md
    drwxr-xr-x 2 root root 4096 Nov 27 10:07 notebooks
    -rw-r--r-- 1 root root   66 Nov 27 10:07 requirements.txt
    drwxr-xr-x 2 root root 4096 Nov 27 10:07 src
    drwxr-xr-x 2 root root 4096 Nov 27 10:07 data
    ```

4. Finally paste the following in another cell:
    ```python
    %cd llm_finetunning
    %pwd

    import sys
    sys.path.append('./')
    ```

    Now you should be able to import code directly from the source directory, e.g:

    ```python
    import src.utils as ut

    ut.letter_to_idx('D')
    # should output: 3
    ```



