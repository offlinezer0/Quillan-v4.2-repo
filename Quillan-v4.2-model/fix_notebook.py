import json
import os

notebook_path = r"c:\Users\Admin\Downloads\Quillan-v4.2-repo-main\Quillan-v4.2-repo-main\Quillan model dev.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to find cell by content
def find_cell_index(source_snippet):
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if source_snippet in source:
                return i
    return -1

# 1. Fix Git Clone
idx = find_cell_index("!git clone https://github.com/leeex1/Quillan-v4.2-repo.git")
if idx != -1:
    print(f"Fixing git clone at cell {idx}")
    source = nb['cells'][idx]['source']
    new_source = []
    for line in source:
        if "!git clone" in line:
            new_source.append("# " + line)
        else:
            new_source.append(line)
    nb['cells'][idx]['source'] = new_source

# 2. Fix Paths (Cell 30 approx)
idx = find_cell_index("jsonl_file_path = \"/content/Quillan-v4.2-repo/Quillan-v4.2-model/Quillan_finetune_full_dataset.jsonl\"")
if idx != -1:
    print(f"Fixing paths at cell {idx}")
    source = nb['cells'][idx]['source']
    new_source = []
    for line in source:
        if "jsonl_file_path =" in line:
            new_source.append("    jsonl_file_path = \"Quillan-v4.2-model/Quillan_finetune_full_dataset.jsonl\"\n")
        elif "quillan_files_dir =" in line:
            new_source.append("    quillan_files_dir = \"Quillan files\"\n")
        else:
            new_source.append(line)
    nb['cells'][idx]['source'] = new_source

# 3. Fix Config Path (Cell 68 approx)
idx = find_cell_index("data_file = '/content/Quillan-v4.2-repo/Quillan-v4.2-model/Quillan_finetune_full_dataset.jsonl'")
if idx != -1:
    print(f"Fixing config path at cell {idx}")
    source = nb['cells'][idx]['source']
    new_source = []
    for line in source:
        if "data_file =" in line:
            new_source.append("    data_file = 'Quillan-v4.2-model/Quillan_finetune_full_dataset.jsonl'  # Flexible path\n")
        else:
            new_source.append(line)
    nb['cells'][idx]['source'] = new_source

# 4. Fix Value class __rmul__ (Cell 68 approx)
idx = find_cell_index("class Value:")
if idx != -1:
    print(f"Fixing Value class at cell {idx}")
    source = nb['cells'][idx]['source']
    new_source = []
    inserted = False
    for line in source:
        new_source.append(line)
        if "return out" in line and "def __mul__" in "".join(source):
             # This is a bit risky relying on exact line, let's look for the end of __mul__ or just add it to the class
             pass
    
    # Easier approach: Replace the whole class definition or insert after __mul__
    # Let's find __mul__ and insert after it
    final_source = []
    for i, line in enumerate(source):
        final_source.append(line)
        if "def __mul__(self, other):" in line:
            # skip to end of method? No, let's just add __rmul__ at the end of the class or after __mul__
            pass
            
    # Let's just append it after __mul__ block. 
    # Actually, let's just find the line "        return out" inside __mul__ and append after that?
    # Too brittle.
    # Let's look for "    def __pow__(self, other):" and insert before it.
    
    final_source = []
    for line in source:
        if "def __pow__(self, other):" in line:
            final_source.append("    def __rmul__(self, other):\n")
            final_source.append("        return self * other\n")
            final_source.append("\n")
        final_source.append(line)
    nb['cells'][idx]['source'] = final_source

# 5. Fix QuillanMoENet __call__ (Cell 68 approx)
idx = find_cell_index("class QuillanMoENet:")
if idx != -1:
    print(f"Fixing QuillanMoENet at cell {idx}")
    source = nb['cells'][idx]['source']
    new_source = []
    skip = False
    for line in source:
        if "def __call__(self, x):" in line:
            new_source.append(line)
            new_source.append("        # Check if input is already a list of Values to avoid double wrapping\n")
            new_source.append("        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], Value):\n")
            new_source.append("            out = x\n")
            new_source.append("        else:\n")
            new_source.append("            out = [Value(xi) for xi in x]\n")
            skip = True # Skip the next line which is the old implementation
        elif skip:
            if "out = [Value(xi) for xi in x]" in line:
                skip = False
            else:
                # If we didn't find the exact line to skip, we might be in trouble. 
                # But based on the file content, it is the very next line.
                # Let's be safer: if we are skipping, and we see "for meta in self.meta_layers:", we stop skipping
                if "for meta in self.meta_layers:" in line:
                    skip = False
                    new_source.append(line)
        else:
            new_source.append(line)
    nb['cells'][idx]['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook fixed successfully.")
