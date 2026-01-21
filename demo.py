# create_dict.py
from pathlib import Path

# Read all labels from train/val/test
all_chars = set()

for split in ['train', 'val', 'test']:
    label_file = Path(f'dataset/{split}/{split}_list.txt')
    
    if label_file.exists():
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    _, label = line.strip().split('\t', 1)
                    all_chars.update(label)

# Sort characters
sorted_chars = sorted(list(all_chars))

# Save dictionary
dict_file = Path('dataset/dict.txt')
with open(dict_file, 'w', encoding='utf-8') as f:
    for char in sorted_chars:
        f.write(f"{char}\n")

print(f"✅ Dictionary created: {dict_file}")
print(f"   Total characters: {len(sorted_chars)}")
print(f"   Characters: {' '.join(sorted_chars)}")