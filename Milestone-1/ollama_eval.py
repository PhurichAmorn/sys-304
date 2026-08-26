import sys
import subprocess

# Ensure ollama library is installed
try:
    import ollama
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
    import ollama

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load train dataset
df = pd.read_csv('/Users/phurich/Documents/CMKL/Fall_2026/SYS-304_Scalable_Algorithms_and_Infrastructure/Milestone-1/nlp-getting-started/train.csv')

# Take the exact same random sample of 50 tweets for direct comparison
sample_df = df.sample(50, random_state=42).copy()

system_prompt = (
    "You are an automated disaster classifier. Classify the tweet as 1 (real disaster, accident, or hazard) or 0 (not a disaster, including figurative speech, jokes, or movie talk). "
    "Analyze the keyword and the tweet text carefully. Respond with exactly one character: '0' or '1'."
)

predictions = []
actuals = list(sample_df['target'])
tweets = list(sample_df['text'])
keywords = list(sample_df['keyword'])

print("Starting few-shot evaluation on 50 sample tweets using qwen2.5:1.5b...")

for idx, tweet in enumerate(tweets):
    kw_str = str(keywords[idx]) if pd.notnull(keywords[idx]) else "None"
    
    # Build a few-shot prompt with examples and keyword metadata
    few_shot_prompt = (
        "Here are examples of how to classify:\n\n"
        "Keyword: fatalities\n"
        "Tweet: \"As of the 6-month mark there were a total of 662 fatalities.\"\n"
        "Class: 1\n\n"
        "Keyword: tragedy\n"
        "Tweet: \"Can't find my ariana grande shirt this is a fucking tragedy...\"\n"
        "Class: 0\n\n"
        "Keyword: sinking\n"
        "Tweet: \"We walk the plank of a sinking ship...\"\n"
        "Class: 0\n\n"
        "Keyword: earthquake\n"
        "Tweet: \"Nepal earthquake 3 months on: Women fear abuse.\"\n"
        "Class: 1\n\n"
        "Now classify the following tweet:\n\n"
        f"Keyword: {kw_str}\n"
        f"Tweet: \"{tweet}\"\n"
        "Class:"
    )
    
    try:
        response = ollama.chat(
            model='qwen2.5:1.5b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': few_shot_prompt}
            ],
            options={
                'temperature': 0.0,
                'num_predict': 2
            }
        )
        pred_text = response['message']['content'].strip()
        
        # Parse prediction
        if '1' in pred_text:
            pred = 1
        elif '0' in pred_text:
            pred = 0
        else:
            pred = 0  # fallback
            
        predictions.append(pred)
        print(f"[{idx+1}/50] Actual: {actuals[idx]} | Pred: {pred} | Keyword: {kw_str} | Tweet: {tweet[:50]}...")
    except Exception as e:
        print(f"[{idx+1}/50] Error predicting: {e}")
        predictions.append(0)

# Compute metrics
print("\n=== FEW-SHOT EVALUATION REPORT ===")
print("Accuracy:", accuracy_score(actuals, predictions))

print("\nConfusion Matrix:")
cm = confusion_matrix(actuals, predictions)
print(f"True Negative (Non-Disaster correctly predicted): {cm[0][0]}")
print(f"False Positive (Non-Disaster predicted as Disaster): {cm[0][1]}")
print(f"False Negative (Disaster predicted as Non-Disaster): {cm[1][0]}")
print(f"True Positive (Disaster correctly predicted): {cm[1][1]}")
print(f"\nRaw Matrix:\n{cm}")

print("\nClassification Report:")
print(classification_report(actuals, predictions, target_names=["Non-Disaster", "Disaster"]))
