"""Small helper to update values in a YAML config without discarding its comments."""

import re


def update_config(path, values):
    """Replace `key: value` lines in place for every key in `values`.

    Only keys that already exist in the file are updated; their comments and
    the rest of the file are left untouched.
    """
    with open(path, "r") as f:
        text = f.read()
    for key, value in values.items():
        pattern = rf"^(\s*{re.escape(key)}:\s*)([^#\n]*?)(\s*(#.*)?)$"
        text, n = re.subn(pattern, lambda m: f"{m.group(1)}{value}{m.group(3)}", text,
                          count=1, flags=re.M)
        if n == 0:
            raise KeyError(f"'{key}' not found in {path}")
    with open(path, "w") as f:
        f.write(text)
    print(f"Updated {', '.join(values)} in {path}")
