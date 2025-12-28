import json

# 请修改为你的 train.json 路径
JSON_FILE = r"F:\bolt_detection_pro\data\annotations\train.json"

with open(JSON_FILE, 'r') as f:
    data = json.load(f)

categories = data.get('categories', [])
print("🔍 正在检查类别 ID...")
print("-" * 30)
for cat in categories:
    print(f"ID: {cat['id']} | Name: {cat['name']}")
print("-" * 30)

if any(cat['id'] == 0 for cat in categories):
    print("❌ 严重警告：发现了 ID 为 0 的类别！")
    print("👉 这会导致 PyTorch 把它当成背景！")
else:
    print("✅ 类别 ID 正常 (没有从 0 开始)。")