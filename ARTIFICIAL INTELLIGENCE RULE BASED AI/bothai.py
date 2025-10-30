import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# both of the csvs came from kaggle
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

fake_df = fake_df[["title"]].dropna()
real_df = real_df[["title"]].dropna()

# Label the data
fake_df["label"] = 1
real_df["label"] = 0

# Combine and shuffle
df = pd.concat([fake_df, real_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset loaded: {len(df)} headlines ({df['label'].sum()} fake, {len(df)-df['label'].sum()} real)")

# AI 1, keyword based ai
fake_keywords = [
    "shocking", "unbelievable", "you won’t believe", "incredible",
    "miracle", "secret", "amazing", "this trick", "shocker", "breaking"
]

def rule_based_ai_1(headline):
    headline_lower = str(headline).lower()
    for word in fake_keywords:
        if word in headline_lower:
            return 1
    return 0

# AI 2, logic based
credible_keywords = [
    "report", "study", "research", "official", "announces", "confirms",
    "data", "statistics", "survey", "evidence", "statement", "agency"
]

def rule_based_ai_2(headline):
    headline_lower = str(headline).lower()
    for word in credible_keywords:
        if word in headline_lower:
            return 0
    return 1

# applying both models
df["AI1_predicted"] = df["title"].apply(rule_based_ai_1)
df["AI2_predicted"] = df["title"].apply(rule_based_ai_2)

# evaluate the models
def evaluate(true, pred):
    return [
        accuracy_score(true, pred),
        precision_score(true, pred),
        recall_score(true, pred),
        f1_score(true, pred)
    ]

ai1_scores = evaluate(df["label"], df["AI1_predicted"])
ai2_scores = evaluate(df["label"], df["AI2_predicted"])

print("\nPerformance Comparison of Rule-Based AIs on Real Dataset")
print("----------------------------------------------------------")
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
print("Metric        | AI #1 (Keyword) | AI #2 (Logic)")
for m, s1, s2 in zip(metrics, ai1_scores, ai2_scores):
    print(f"{m:<13} | {s1:.3f}          | {s2:.3f}")

# plot comparison 
x = range(len(metrics))
plt.figure(figsize=(8,5))
plt.bar([i - 0.2 for i in x], ai1_scores, width=0.4, label='AI #1: Keyword-Based')
plt.bar([i + 0.2 for i in x], ai2_scores, width=0.4, label='AI #2: Logic-Based')

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Fake News Detection: Rule-Based AI Comparison (Real Dataset)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
