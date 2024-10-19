import os

# TODO: zaiko packegeに書かれたgoのコードによりinitを行う
script = f"""\

"""

target = os.path.join(os.path.dirname(__file__), "entrypoint.sh")

with open(target, 'w') as file:
    file.write(script)

print(f"[zaiko] entrypoint.sh has been written to {target}.")
