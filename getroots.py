# save as tree_view.py and run: python tree_view.py
import os

def print_tree(startpath, prefix="", depth=0, max_depth=2):
    if depth > max_depth:
        return

    try:
        entries = os.listdir(startpath)
    except PermissionError:
        return

    files = []
    dirs = []
    for entry in entries:
        full_path = os.path.join(startpath, entry)
        if os.path.isdir(full_path):
            dirs.append(entry)
        else:
            files.append(entry)

    # Print files
    for f in files:
        print(prefix + "├── " + f)

    # Print directories and recurse
    for d in dirs:
        print(prefix + "├── " + d + "/")
        print_tree(os.path.join(startpath, d), prefix + "│   ", depth + 1, max_depth)

if __name__ == "__main__":
    root = "."
    max_depth = 2   # 👈 change this number to control depth
    print(f"Project structure of: {os.path.abspath(root)}\n")
    print_tree(root, max_depth=max_depth)
