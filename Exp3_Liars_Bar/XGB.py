import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
from xgboost import plot_importance
from lime.lime_tabular import LimeTabularExplainer

# 读取数据
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "all_plays_with_challenge_info.csv")

df = pd.read_csv(csv_path)

# 删除不需要的列（包括 'player', 'challenger', 'challenge_thinking', 'play_thinking', 'game_id'）
df = df.drop(columns=[
    "game_id", "player", "challenger", 
    "challenge_thinking", "play_thinking","challenge_result"
], errors="ignore")

df["was_challenged"] = df["was_challenged"].astype(bool)
df["previous_successful_challenge"] = df["previous_successful_challenge"].astype(bool)

# 设置目标变量
y = df["was_challenged"].astype(int)

# 特征
X = df.drop(columns=["was_challenged"])

# 编码类别特征（这里只剩下 target_card 是字符串）
if "target_card" in X.columns:
    le = LabelEncoder()
    X["target_card"] = le.fit_transform(X["target_card"].astype(str))

# 填充缺失值（如 None -> -1）
X = X.fillna(-1)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43, stratify=y
)

# 训练模型
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# 模型预测与评估
y_pred = model.predict(X_test)
print("📊 分类报告：\n")
print(classification_report(y_test, y_pred))
plot_importance(model)
plt.tight_layout()
plt.show()

# 创建 LIME 解释器
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Not Challenged", "Challenged"],
    mode="classification"
)

# 选择解释测试集中的第 i 条数据
i = 0  # 可改为任意 index（范围：0 ~ len(X_test)-1）
instance = X_test.iloc[i].values

# 获取解释结果
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba,
    num_features=10  # 可改为任意你想看的特征数量
)

# 在 notebook 中展示（或弹出浏览器）
exp.show_in_notebook(show_table=True)