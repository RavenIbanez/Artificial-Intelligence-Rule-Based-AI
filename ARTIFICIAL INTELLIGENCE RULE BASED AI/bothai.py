import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# Keep only the titles and assign labels (1 = fake, 0 = real)
fake_df = fake_df[["title"]].dropna()
real_df = real_df[["title"]].dropna()
fake_df["label"] = 1
real_df["label"] = 0

# Combine and shuffle
df = pd.concat([fake_df, real_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Dataset loaded: {len(df)} headlines ({df['label'].sum()} fake, {len(df)-df['label'].sum()} real)")

# AI 1: clickbait/emotional indicator words (detects fake news)
fake_keywords = [
    "shocking", "unbelievable", "you won’t believe", "incredible", "miracle", "secret", "amazing", "this trick",
    "shocker", "breaking", "disaster", "warning", "urgent", "bombshell", "scandal", "truth revealed", "lies exposed",
    "exclusive", "must see", "bizarre", "outrage", "controversial", "terrifying", "mind-blowing", "crazy", "viral",
    "spotted", "rumor", "fake", "hoax", "cover up", "caught on camera", "gone wrong", "explosive", "mystery", "shame"
]

def rule_based_ai_1(headline):
    headline_lower = str(headline).lower()
    for word in fake_keywords:
        if word in headline_lower:
            return 1
    return 0

# AI 2: factual/credible indicator words (detects real news)
credible_keywords = [
    "report", "study", "research", "official", "announces", "confirms", "data", "statistics", "survey",
    "evidence", "statement", "agency", "experts", "analysis", "investigation", "according", "reportedly",
    "findings", "minister", "organization", "scientists", "professor", "university", "results", "reporter",
    "update", "spokesperson", "publication", "conference", "press release", "verified", "confirmed", "figures"
]

def rule_based_ai_2(headline):
    headline_lower = str(headline).lower()
    for word in credible_keywords:
        if word in headline_lower:
            return 0
    return 1

# Apply both AIs to the dataset
df["AI1_predicted"] = df["title"].apply(rule_based_ai_1)
df["AI2_predicted"] = df["title"].apply(rule_based_ai_2)

# Basic fake/real counts
ai1_fake = (df["AI1_predicted"] == 1).sum()
ai1_real = (df["AI1_predicted"] == 0).sum()

ai2_fake = (df["AI2_predicted"] == 1).sum()
ai2_real = (df["AI2_predicted"] == 0).sum()

comparison_table = pd.DataFrame({
    "Prediction": ["Fake", "Real"],
    "AI 1 (Keyword-Based)": [ai1_fake, ai1_real],
    "AI 2 (Logic-Based)": [ai2_fake, ai2_real]
})

print("\nFake vs Real Predictions Comparison")
print("-----------------------------------")
print(comparison_table.to_string(index=False))

# Chart 1: Fake vs Real counts
plt.figure(figsize=(7,5))
x = range(2)
plt.bar([i - 0.2 for i in x], [ai1_fake, ai1_real], width=0.4, label="AI 1: Keyword-Based")
plt.bar([i + 0.2 for i in x], [ai2_fake, ai2_real], width=0.4, label="AI 2: Logic-Based")
plt.xticks(x, ["Fake", "Real"])
plt.ylabel("Count")
plt.title("Fake vs Real Predictions by AI 1 and AI 2")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# Evaluate both models
def evaluate(true, pred):
    return [
        accuracy_score(true, pred),
        precision_score(true, pred),
        recall_score(true, pred),
        f1_score(true, pred)
    ]

ai1_scores = evaluate(df["label"], df["AI1_predicted"])
ai2_scores = evaluate(df["label"], df["AI2_predicted"])

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
results_df = pd.DataFrame({
    "Metric": metrics,
    "AI 1 (Keyword-Based)": ai1_scores,
    "AI 2 (Logic-Based)": ai2_scores
})

print("\nPerformance Comparison of Rule-Based AIs on Real Dataset")
print("----------------------------------------------------------")
print(results_df.to_string(index=False, float_format="%.3f"))

# Chart 2: Performance metrics
plt.figure(figsize=(8,5))
x = range(len(metrics))
plt.bar([i - 0.2 for i in x], ai1_scores, width=0.4, label="AI 1: Keyword-Based")
plt.bar([i + 0.2 for i in x], ai2_scores, width=0.4, label="AI 2: Logic-Based")
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Performance Metrics Comparison of Rule-Based AIs")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# Find disagreement cases between AI 1 and AI 2
disagreements = df[df["AI1_predicted"] != df["AI2_predicted"]]

print("\nAI Disagreement Examples (randomly sampled):")
print("--------------------------------------------")

# When AI 1 says fake but AI 2 says real
ai1_fake_ai2_real = disagreements[(disagreements["AI1_predicted"] == 1) & (disagreements["AI2_predicted"] == 0)]
if not ai1_fake_ai2_real.empty:
    print("\nAI 1 marked as Fake, AI 2 marked as Real:")
    print(ai1_fake_ai2_real["title"].sample(min(5, len(ai1_fake_ai2_real)), random_state=1).to_string(index=False))

# When AI 1 says real but AI 2 says fake
ai1_real_ai2_fake = disagreements[(disagreements["AI1_predicted"] == 0) & (disagreements["AI2_predicted"] == 1)]
if not ai1_real_ai2_fake.empty:
    print("\nAI 1 marked as Real, AI 2 marked as Fake:")
    print(ai1_real_ai2_fake["title"].sample(min(5, len(ai1_real_ai2_fake)), random_state=2).to_string(index=False))
