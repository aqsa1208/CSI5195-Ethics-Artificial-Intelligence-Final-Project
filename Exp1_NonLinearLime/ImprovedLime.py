import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
#from xgboost import XGBRegressor
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from lime.lime_text import LimeTextExplainer,  IndexedString
import matplotlib.pyplot as plt
from collections import defaultdict
from lime.lime_base import LimeBase
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import pairwise_distances
from functools import lru_cache
from pathlib import Path
import gc
import math
from itertools import product





# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n=== System Initialization ===")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# === Data Loading ===
def load_data():
    data_dir = 'sentiment labelled sentences'
    dfs = []
    for filename in ['amazon_cells', 'imdb', 'yelp']:
        path = os.path.join(data_dir, f'{filename}_labelled.txt')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file {path} not found")
        df = pd.read_csv(path, sep='\t', header=None, names=['text', 'label'], on_bad_lines='warn')
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Data loaded. Total samples: {len(full_df)}")
    return full_df

# Load data only once
df = load_data()
#  train / val / test
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['text'], df['label'], test_size=0.3, stratify=df['label'], random_state=42)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.3, stratify=temp_labels, random_state=42)

model_path = Path("./fine_tuned_model")
model_files_exist = (model_path / "model.safetensors").exists() and (model_path / "config.json").exists()
tokenizer_files_exist = (model_path / "vocab.txt").exists() and (model_path / "tokenizer_config.json").exists()

# === Tokenizer and Dataset ===
#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
if tokenizer_files_exist:
    tokenizer = DistilBertTokenizer.from_pretrained(str(model_path))
else:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist())
test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist())
val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist())

# === Model and Training ===
#model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

if model_path.exists() and model_path.is_dir() and model_files_exist and tokenizer_files_exist:
    print("Loading existing model and tokenizer...")
    model = DistilBertForSequenceClassification.from_pretrained(str(model_path)).to(device)
else:
    print(f"No existing model found. Starting training...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        save_strategy="no",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    print("Model and tokenizer saved to:", model_path)

def prob_to_logodds(p):
    return np.log(p / (1 - p + 1e-9) + 1e-9)

# === Explanation System ===
class NonLinearLimeExplainer:
    def __init__(self, class_names, num_samples=5000, kernel_width=25, distance_metric='cosine', split_expression=r'\W+',bow=True, random_state=42):
        self.class_names = class_names
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.distance_metric = distance_metric
        self.bow = bow
        self.random_state = np.random.RandomState(random_state)

        self.split_expression = split_expression

        self.kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / self.kernel_width ** 2))
        self.base = LimeBase(kernel_fn=self.kernel_fn)
        self.model_regressor = GradientBoostingRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42
        )

    def explain_instance(self,
                         text_instance,
                         classifier_fn,
                         labels=None,
                         top_labels=None,
                         num_features=10,
                         return_details=False,
                         external_data=None):

        fallback_mode = False

        if external_data is not None:
            data, predictions, indexed_string = external_data
            num_tokens = data.shape[1]
            if num_tokens < 4:
                #num_tokens < 5
                #2 ** num_tokens * 10 < self.num_samples
                #print(f"[Fallback] Token count too low ({num_tokens}), using internal perturbation.")
                fallback_mode = True
        else:
            fallback_mode = True

        if fallback_mode:
            indexed_string = IndexedString(
                text_instance,
                split_expression=self.split_expression,
                bow=self.bow,
                mask_string=None
            )
            num_tokens = indexed_string.num_words()

            #data = self.random_state.randint(0, 2, size=(self.num_samples, num_tokens))

            all_states = np.array(list(product([0, 1], repeat=num_tokens)))
            N_unique = all_states.shape[0]
            reps = int(math.ceil(self.num_samples / N_unique))
            expanded_states = np.tile(all_states, (reps, 1))
            total = expanded_states.shape[0]
            indices = self.random_state.choice(total, size=self.num_samples, replace=False)
            data = expanded_states[indices]
            data[0, :] = 1

            # === Step 3: Generate perturbed texts ===
            perturbed_texts = [
                indexed_string.inverse_removing([i for i in range(num_tokens) if row[i] == 0])
                for row in data
            ]

            predictions = classifier_fn(perturbed_texts)

        # === Step 5: Compute distances ===
        distances = pairwise_distances(data, data[0].reshape(1, -1), metric=self.distance_metric).ravel() * 100

        weights = self.kernel_fn(distances)

        # === Step 6: Determine labels to explain ===
        if labels is not None:
            label_list = list(labels)
        elif top_labels is not None:
            label_list = list(np.argsort(predictions[0])[-top_labels:][::-1])
        else:
            label_list = [np.argmax(predictions[0])]

        explanations = []
        for label in label_list:

            #target = predictions[:, label] - predictions[0, label]
            target = predictions[:, label]
            self.model_regressor.fit(data, target, sample_weight=weights)

            importances = self.model_regressor.feature_importances_

            token_list = [indexed_string.word(i) for i in range(num_tokens)]
            #sorted_features = sorted(zip(token_list, importances), key=lambda x: abs(x[1]), reverse=True)
            #exp = [(feat, {'mean': float(score), 'std': 0.0}) for feat, score in sorted_features[:num_features]]
            #explanations.append((label, exp))
            scored_tokens = []
            #original_pred_label = predictions[0, label]

            for i in range(num_tokens):
                token = token_list[i]
                importance = importances[i]
                mask = data[:, i] == 1
                if np.any(mask) and np.any(~mask):
                    avg_keep = np.mean(target[mask])
                    avg_remove = np.mean(target[~mask])
                    direction = avg_keep - avg_remove
                    # direction = np.sign(raw_direction) * (abs(raw_direction / (np.std(target) + 1e-6)) ** 2)
                else:
                    direction = 0.0

                # text_removed_i = indexed_string.inverse_removing([i])
                # pred_removed_i = classifier_fn([text_removed_i])[0][label]
                # direction = original_pred_label - pred_removed_i


                combined_score = importance * direction
                scored_tokens.append((token, direction, combined_score))

            sorted_tokens = sorted(scored_tokens, key=lambda x: abs(x[2]), reverse=True)

            exp = []
            for token, direction, _ in sorted_tokens[:num_features]:
                exp.append((token, {"mean": float(direction), "std": 0.0}))

            explanations.append((label, exp))

        # === Output ===
        if return_details:
            if len(label_list) == 1:
                #target = predictions[:, label_list[0]] - predictions[0, label_list[0]]
                target = predictions[:, label_list[0]]
                return explanations[0][1], data, target
            else:
                target_dict = {
                    #label: predictions[:, label] - predictions[0, label]
                    label: predictions[:, label]
                    for label in label_list
                }
                return (
                    {label: exp for label, exp in explanations},
                    data,
                    target_dict
                )
        else:
            if len(label_list) == 1:
                return explanations[0][1]
            else:
                return {label: exp for label, exp in explanations}


class PredictWrapper:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device

    def predict_proba(self, texts, batch_size=32):
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                results.append(probs)

            torch.cuda.empty_cache()

        return np.vstack(results)

    @lru_cache(maxsize=10000)
    def predict_proba_cached(self, text):
        return self.predict_proba([text])[0]

# === Visualization System ===
def evaluate_explanation_metrics(text, label, predictor, max_k=None, repeats=10, num_features=10, return_stability_dict=False, skip_explanation=False):
    results = {}

    try:
        # === Step 1: 统一生成扰动样本 ===
        indexed = IndexedString(text, bow=False)
        X, y, _ = classic_explainer._LimeTextExplainer__data_labels_distances(
            indexed, predictor.predict_proba, 5000
        )

        # === Step 2: 获取两个解释器的解释 ===
        exp_lime = classic_explainer.explain_instance(text, predictor.predict_proba, labels=(label,))
        lime_items = exp_lime.as_list(label=label)

        exp_nl, X_nl, y_nl = improved_explainer.explain_instance(
            text, predictor.predict_proba,
            labels=(label,),
            return_details=True,
            external_data=(X, y, indexed)
        )
        y_pred_nl = improved_explainer.model_regressor.predict(X_nl)

        # === Step 3: Fidelity ===
        distances = pairwise_distances(X, X[0].reshape(1, -1), metric="cosine").ravel() * 100
        _, _, r2_lime, _ = classic_explainer.base.explain_instance_with_data(X, y, distances, label, num_features)
        r2_nl = r2_score(y_nl, y_pred_nl)
        results["Fidelity_LIME"] = r2_lime
        results["Fidelity_Nonlinear"] = r2_nl
        print(f"Fidelity_NL:{r2_nl}")

        # === Step 4: Faithfulness (AOPC) ===
        orig_prob = predictor.predict_proba_cached(text)

        ranked_lime = [feat for feat, _ in lime_items]
        ranked_nl = [feat for feat, _ in exp_nl]

        if max_k is not None:
            try:
                max_k = int(max_k)
                ranked_lime = ranked_lime[:max_k]
                ranked_nl = ranked_nl[:max_k]
            except:
                pass

        diffs_lime, diffs_nl = [], []

        for i in range(1, len(ranked_lime) + 1):
            masked = text
            for token in ranked_lime[:i]:
                masked = masked.replace(token, '')
            new_prob = predictor.predict_proba_cached(masked)
            diffs_lime.append(orig_prob[label] - new_prob[label])

        for i in range(1, len(ranked_nl) + 1):
            masked = text
            for token in ranked_nl[:i]:
                masked = masked.replace(token, '')
            new_prob = predictor.predict_proba_cached(masked)
            diffs_nl.append(orig_prob[label] - new_prob[label])

        results["Faithfulness_LIME"] = np.mean(diffs_lime)
        results["Faithfulness_Nonlinear"] = np.mean(diffs_nl)
        print(f"Faithfulness_NL:{results["Faithfulness_Nonlinear"]}")

        # === Step 5: Stability ===
        stab_lime = defaultdict(list)
        stab_nl = defaultdict(list)

        for _ in range(repeats):
            try:
                # 每轮新扰动样本
                X_rep, y_rep, _ = classic_explainer._LimeTextExplainer__data_labels_distances(
                    indexed, predictor.predict_proba, 5000
                )

                # LIME
                exp_lime_r = classic_explainer.explain_instance(text, predictor.predict_proba, labels=(label,))
                items_lime = exp_lime_r.as_list(label=label)
                for feat, score in items_lime:
                    stab_lime[feat].append(score)

                # Nonlinear
                exp_nl_r, _, _ = improved_explainer.explain_instance(
                    text, predictor.predict_proba,
                    labels=(label,),
                    return_details=True,
                    external_data=(X_rep, y_rep, indexed)
                )
                for feat, score in exp_nl_r:
                    score = score["mean"] if isinstance(score, dict) else score
                    stab_nl[feat].append(score)

            except Exception as e:
                print(f"[Stability Repeat Error] {e}")
                continue

        stab_std_lime = {k: np.std(v) for k, v in stab_lime.items()}
        stab_std_nl = {k: np.std(v) for k, v in stab_nl.items()}

        if return_stability_dict:
            results["Stability_LIME"] = stab_std_lime
            results["Stability_Nonlinear"] = stab_std_nl
        else:
            results["Stability_LIME"] = np.mean(list(stab_std_lime.values())) if stab_std_lime else 0.0
            results["Stability_Nonlinear"] = np.mean(list(stab_std_nl.values())) if stab_std_nl else 0.0

    except Exception as e:
        print(f"[Evaluation Error] {e}")
        results = {
            "Fidelity_LIME": 0.0,
            "Fidelity_Nonlinear": 0.0,
            "Faithfulness_LIME": 0.0,
            "Faithfulness_Nonlinear": 0.0,
            "Stability_LIME": {} if return_stability_dict else 0.0,
            "Stability_Nonlinear": {} if return_stability_dict else 0.0
        }

    torch.cuda.empty_cache()
    gc.collect()

    if skip_explanation:
        return results
    else:
        return results, X, y, indexed, lime_items, exp_nl


def plot_comparison(text, label=None):
    if label is None:
        probs = predictor.predict_proba([text])[0]
        label = int(np.argmax(probs))

    # === 调用统一评估函数 ===
    metrics, X, y, indexed, lime_items, exp_nl = evaluate_explanation_metrics(
        text, label, predictor, return_stability_dict=True, skip_explanation= False
    )

    # === 解析解释结果 ===
    lime_tokens = [f[0] for f in lime_items]
    lime_weights = [f[1] for f in lime_items]

    nl_tokens = [f[0] for f in exp_nl]
    nl_weights = [f[1]["mean"] for f in exp_nl]

    # === 提取指标值 ===
    r2_lime = metrics["Fidelity_LIME"]
    r2_nl = metrics["Fidelity_Nonlinear"]

    aopc_lime = metrics["Faithfulness_LIME"]
    aopc_nl = metrics["Faithfulness_Nonlinear"]

    stab_lime = metrics["Stability_LIME"]
    stab_nl = metrics["Stability_Nonlinear"]
    print(stab_nl)

    if isinstance(stab_lime, dict):
        tokens_lime_std = list(stab_lime.keys())
        stds_lime = list(stab_lime.values())
    else:
        tokens_lime_std, stds_lime = [], []

    if isinstance(stab_nl, dict):
        tokens_nl_std = list(stab_nl.keys())
        stds_nl = list(stab_nl.values())
    else:
        tokens_nl_std, stds_nl = [], []

    # === 绘图 ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
    axes = axes.flatten()

    # === 0. LIME Explanation ===
    axes[0].barh(lime_tokens[::-1], lime_weights[::-1], color='steelblue')
    axes[0].set_title(f"LIME Explanation (Label {label})")

    # === 1. Nonlinear LIME Explanation ===
    axes[1].barh(nl_tokens[::-1], nl_weights[::-1], color='skyblue')
    axes[1].set_title(f"Nonlinear LIME Explanation (Label {label})")

    # === 2. Fidelity (R²) ===
    axes[2].bar(["LIME", "Nonlinear"], [r2_lime, r2_nl], color=["purple", "skyblue"])
    min_y = min(r2_lime, r2_nl)
    max_y = max(r2_lime, r2_nl)
    if all(val >= 0 for val in [r2_lime, r2_nl]):
        axes[2].set_ylim(0, max_y + 0.2)
    else:
        y_margin = (max_y - min_y) * 0.2 if max_y != min_y else 0.1
        axes[2].set_ylim(min_y - y_margin, max_y + y_margin + 0.2)
    #axes[2].set_ylim(min(0, r2_lime, r2_nl), max(r2_lime, r2_nl) + 0.05)
    axes[2].set_title("Fidelity (R²)")
    for i, val in enumerate([r2_lime, r2_nl]):
        axes[2].text(i, val + 0.02, f"{val:.4f}", ha="center")

    # === 3. LIME Stability ===
    axes[3].barh(tokens_lime_std[::-1], stds_lime[::-1], color='orange')
    axes[3].set_title("LIME Stability (STD)")

    # === 4. Nonlinear LIME Stability ===
    axes[4].barh(tokens_nl_std[::-1], stds_nl[::-1], color='salmon')
    axes[4].set_title("Nonlinear LIME Stability (STD)")

    # === 5. Faithfulness (AOPC) ===
    axes[5].bar(["LIME", "Nonlinear"], [aopc_lime, aopc_nl], color=["green", "lightgreen"])
    axes[5].set_ylim(0, max(aopc_lime, aopc_nl) + 0.05)
    axes[5].set_title("Faithfulness (AOPC)")
    for i, val in enumerate([aopc_lime, aopc_nl]):
        axes[5].text(i, val + 0.01, f"{val:.4f}", ha="center")

    # === 坐标轴范围统一处理 ===
    all_weights = lime_weights + nl_weights
    if all_weights:
        x_limit_exp = max(abs(min(all_weights)), abs(max(all_weights))) + 0.1
        axes[0].set_xlim(-x_limit_exp, x_limit_exp)
        axes[1].set_xlim(-x_limit_exp, x_limit_exp)

    all_stds = stds_lime + stds_nl
    if all_stds:
        x_limit_std = max(all_stds) + 0.05
        axes[3].set_xlim(0, x_limit_std)
        axes[4].set_xlim(0, x_limit_std)

    plt.suptitle(f"Explanation Comparison for Text: {text[:80]}...", fontsize=14)
    plt.tight_layout()
    plt.show()



# === Explanation Execution ===
predictor = PredictWrapper(model, tokenizer, device=device)
classic_explainer = LimeTextExplainer(class_names=['negative', 'positive'], bow=False,random_state=42)
improved_explainer = NonLinearLimeExplainer(class_names=['negative', 'positive'], bow=False,random_state=42)



def run_analysis(num_samples=4):
    np.random.seed(None)
    samples = np.random.choice(test_texts.tolist(), num_samples, replace=False)
    for text in samples:
        probs = predictor.predict_proba([text])[0]
        predicted_label = int(np.argmax(probs))
        print(f"\nAnalyzing: {text}...")
        print(f"Predicted label: {predicted_label} ({improved_explainer.class_names[predicted_label]})")
        plot_comparison(text,predicted_label)

def evaluate_all(save_path="lime_metrics_results.csv"):
    sampled_texts = test_texts.tolist()
    num_samples = len(sampled_texts)

    save_path = Path(save_path)

    # === Step 1: 如果已有文件，保留当前测试集中的历史记录 ===
    if save_path.exists():
        df_existing = pd.read_csv(save_path)
        df_filtered = df_existing[df_existing["Text"].isin(sampled_texts)].copy()
        existing_texts = set(df_filtered["Text"])
    else:
        df_filtered = pd.DataFrame()
        existing_texts = set()

    # === Step 2: 遍历新样本 ===
    for idx, text in enumerate(sampled_texts):
        if text in existing_texts:
            continue

        try:
            probs = predictor.predict_proba([text])[0]
            label = int(np.argmax(probs))

            #  使用统一评估函数（只调用一次）
            metrics = evaluate_explanation_metrics(
                text, label, predictor, return_stability_dict=False,skip_explanation = True
            )

            # === 构建新行并追加 ===
            new_row = {
                "Text": text,
                "Label": label,
                "Fidelity_LIME": metrics["Fidelity_LIME"],
                "Fidelity_Nonlinear": metrics["Fidelity_Nonlinear"],
                "Faithfulness_LIME": metrics["Faithfulness_LIME"],
                "Faithfulness_Nonlinear": metrics["Faithfulness_Nonlinear"],
                "Stability_LIME": metrics["Stability_LIME"],
                "Stability_Nonlinear": metrics["Stability_Nonlinear"]
            }
            write_header = not save_path.exists()
            pd.DataFrame([new_row]).to_csv(
                save_path,
                mode='a',
                header=write_header,
                index=False
            )


            print(f"[{idx+1}/{num_samples}] Evaluating: {text[:60]}... Saved.")


        except RuntimeError as e:
            print(f"[{idx+1}/{num_samples}] Skipped due to error: {e}")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[{idx+1}/{num_samples}] General error: {e}")
            torch.cuda.empty_cache()
            continue
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nEvaluation complete. Results saved to: {save_path}")
    return pd.read_csv(save_path)



def plot_metrics_comparison(file_path="lime_metrics_results.csv"):
    if os.path.exists(file_path):
        print(f"[INFO] Found existing file: {file_path}, loading results...")
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx.")

    else:
        print(f"[INFO] No file found at {file_path}, running evaluation...")
        df = evaluate_all(save_path=file_path)


    metric_names = ["Fidelity", "Faithfulness", "Stability"]
    explainers = ["LIME", "Nonlinear"]


    metrics_mean = {
        metric: {
            explainer: df[f"{metric}_{explainer}"].mean()
            for explainer in explainers
        }
        for metric in metric_names
    }

    x = np.arange(len(metric_names))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=100)

    for i, metric in enumerate(metric_names):
        lime_val = metrics_mean[metric]["LIME"]
        nonlinear_val = metrics_mean[metric]["Nonlinear"]
        ax = axes[i]
        ax.bar(x[i] - width/2, lime_val, width, label='LIME', color='steelblue', alpha=0.8)
        ax.bar(x[i] + width/2, nonlinear_val, width, label='Nonlinear LIME', color='skyblue', alpha=0.8)

        ax.set_title(f"{metric}", fontsize=13)
        ax.set_ylabel("Score")
        ax.set_xticks([x[i]])
        ax.set_xticklabels([metric])
        #ax.set_ylim(0, max(lime_means[i], nonlinear_means[i]) + 0.2)

        #for j, val in enumerate([lime_means[i], nonlinear_means[i]]):
        #    xpos = x[i] - width/2 if j == 0 else x[i] + width/2
        #    ax.text(xpos, val + 0.02, f"{val:.4f}", ha='center', fontsize=10)
        min_y = min(lime_val, nonlinear_val)
        max_y = max(lime_val, nonlinear_val)

        # 如果两个分数都是非负，则从 0 开始
        if all(val >= 0 for val in [lime_val, nonlinear_val]):
            ax.set_ylim(0, max_y + 0.2)
        else:
            y_margin = (max_y - min_y) * 0.2 if max_y != min_y else 0.1
            ax.set_ylim(min_y - y_margin, max_y + y_margin + 0.2)

        for j, val in enumerate([lime_val, nonlinear_val]):
            xpos = x[i] - width / 2 if j == 0 else x[i] + width / 2
            y_offset = 0.02 if val >= 0 else -0.05
            ax.text(xpos, val + y_offset, f"{val:.4f}", ha='center', fontsize=10)

    axes[0].legend(loc='upper left')
    plt.suptitle("Metric Comparison Across All Test Examples", fontsize=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mode = input("Enter '1' for single example or '2' for full evaluation: ")
    if mode.strip() == '1':
        #run_analysis(num_samples=4)
        plot_comparison('Excellent!.',1)
    else:
        file_path = "results/metrics_comparison_2025(14).csv"
        metrics = evaluate_all(file_path)
        plot_metrics_comparison(file_path)