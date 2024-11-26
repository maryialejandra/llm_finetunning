import os

def get_secret(var_name: str,  value: str = None,
               use_env: bool=True,
               use_colab: bool=True):
    if value is not None:
        print(f"Returning secret from passed argument: {value[:2]}...{value[-2:]}", )
        return value

    if use_env:
        value = os.environ.get(var_name)
        if value is not None:
            print(f"Returning secret from environment variable `{var_name}`=`{value[:2]}...{value[-2:]}`", )
            return value
        else:
            print(f"Secret {var_name} not found in env_var")

    if use_colab:
        from google.colab import userdata
        value = userdata.get(var_name)
        if value is not None:
            print(f"Returning google.colab secret `{var_name}`=`{value[:2]}...{value[-2:]}`", )
        else:
            print(f"Secret {var_name} not found in google.colab secrets")

    if value is None:
        raise ValueError(f"Secret for {var_name} not found! "
                         f"try switching one of the following flags to true use_env={use_env} use_colab={use_colab}")
