import pandas as pd
import time
from openai import OpenAI
import os

# ✅ 替换为你的 OpenAI API key
client = OpenAI(api_key="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") 
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "all_plays_with_challenge_info.csv")

# 读取 CSV 文件
df = pd.read_csv(csv_path)

# 确保存在 opinion_text 列
if "opinion_text" not in df.columns:
    raise ValueError("CSV 文件中缺少 'opinion_text' 列，请检查文件结构。")

# 标签提示模板
LABEL_PROMPT = """
你是一名策略博弈分析专家，以下是玩家对另一个玩家行为的看法，请你根据描述判断其属于哪种博弈风格。你只能从以下六种类型中选择一个作为分类标签，并且只输出标签本身，不要输出解释。

可选标签为：
1. 保守型（Conservative）
2. 激进型（Aggressive）
3. 策略型（Strategic）
4. 情绪化（Reactive）
5. 混合型（Hybrid）
6. 未知（Unknown）

描述如下：
"{text}"

请根据上面的描述输出分类标签：
"""

# 创建一个新列用于存储分类结果
df["opinion_label"] = ""

# 分类函数
def classify_opinion(text):
    if pd.isna(text) or text.strip() == "":
        return "未知"
    
    prompt = LABEL_PROMPT.format(text=text.strip())

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个善于分析博弈行为的专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        reply = response.choices[0].message.content.strip()
        return reply
    except Exception as e:
        print(f"❌ 处理文本失败：{e}")
        return "未知"

# 遍历并分类（建议小批量测试）
for idx, row in df.iterrows():
    opinion = row["opinion_text"]
    print(f"🧠 正在分析第 {idx + 1} 条文本...")
    label = classify_opinion(opinion)
    df.at[idx, "opinion_label"] = label

# 保存输出
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("✅ 已保存分类结果至 all_plays_with_opinion_labels.csv")
