import os

EXCLUDE = {
    ".env",
    ".gitignore",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "tree.py"  # ← se excluye a sí mismo
}

def print_tree(start_path, indent=""):
    items = sorted(os.listdir(start_path))
    items = [item for item in items if item not in EXCLUDE]

    for index, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = index == len(items) - 1

        prefix = "└── " if is_last else "├── "
        print(indent + prefix + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, indent + extension)

print_tree(".")
