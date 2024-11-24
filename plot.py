#%%
import pandas as pd
import matplotlib.pyplot as plt
import re

# Function to parse the log data
def parse_log_entry(entry):
    match = re.search(r"Epoch: (\d+), Eval Score: ([\d.]+), PROMPT: (.*?), TARGET_BEHAVIOR: (.*?), num_features = (\d+)", entry)
    if match:
        epoch = int(match.group(1))
        eval_score = float(match.group(2))
        prompt = match.group(3)
        target_behavior = match.group(4)
        num_features = int(match.group(5))
        return epoch, eval_score, prompt, target_behavior, num_features
    return None

# Read and parse log file
with open('epoch_eval_logs.txt', 'r') as file:
    log_entries = file.readlines()
    parsed_data = [parse_log_entry(entry) for entry in log_entries]
    parsed_data = [data for data in parsed_data if data is not None]

    # Creating a DataFrame
    df = pd.DataFrame(parsed_data, columns=["Epoch", "Eval Score", "Prompt", "Target Behavior", "Num Features"])

    # Summarizing each prompt
    df["Prompt Summary"] = df["Prompt"].apply(lambda x: " ".join(x.split()[:5]) + "...")

#%%
# Create three subplots for different target behaviors
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
axes = [ax1, ax2, ax3]

# Filter for num_features = 5
df_filtered = df[df["Num Features"] == 5]

# Get unique target behaviors
target_behaviors = df_filtered["Target Behavior"].unique()

# Plot each target behavior in a separate subplot
for idx, target in enumerate(target_behaviors):  # limit to 3 target behaviors
    df_target = df_filtered[df_filtered["Target Behavior"] == target]
    
    for prompt, group in df_target.groupby("Prompt Summary"):
        axes[idx].plot(group["Epoch"], group["Eval Score"], label=prompt, marker='o')
    
    axes[idx].set_xlabel("Epoch")
    axes[idx].set_ylabel("Eval Score")
    axes[idx].set_title(f"Evaluation Score vs. Epoch ({target})")
    axes[idx].legend(title="Prompt Summary", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[idx].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig('target_behavior_comparison.png', bbox_inches='tight')
# Create a figure for each unique target behavior
for target in target_behaviors:
    plt.figure(figsize=(10, 6))
    df_target = df_filtered[df_filtered["Target Behavior"] == target]
    
    for prompt, group in df_target.groupby("Prompt Summary"):
        plt.plot(group["Epoch"], group["Eval Score"], label=prompt, marker='o')
    
    plt.xlabel("Epoch")
    plt.ylabel("Eval Score")
    plt.title(f"Evaluation Score vs. Epoch ({target})")
    plt.legend(title="Prompt Summary", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f'target_behavior_{target.replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()

#%%
# for the same target_behavior, plot the evaluation score vs. epoch for each prompt, one graph for each num_features value, and save the plots as separate files, set tilte to reflect the num_features value
for target in target_behaviors:
    df_target = df[df["Target Behavior"] == target]
    num_features_values = df_target["Num Features"].unique()
    
    for num_features in num_features_values:
        plt.figure(figsize=(10, 6))
        df_num_features = df_target[df_target["Num Features"] == num_features]
        
        for prompt, group in df_num_features.groupby("Prompt Summary"):
            plt.plot(group["Epoch"], group["Eval Score"], label=prompt, marker='o')
        
        plt.xlabel("Epoch")
        plt.ylabel("Eval Score")
        plt.title(f"Evaluation Score vs. Epoch ({target}) - Num Features: {num_features}")
        plt.legend(title="Prompt Summary", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(f'target_behavior_{target.replace(" ", "_")}_num_features_{num_features}.png', bbox_inches='tight')
        plt.close()
#%%
# save df to file
# df.to_csv("parsed_epoch_eval_logs_params_targetb_numfeats.csv", index=False)
# %%
