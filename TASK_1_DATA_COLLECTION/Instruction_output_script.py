import pandas as pd
import random
from google.colab import files
from unsloth import to_sharegpt

# Step 1: Upload your dataset manually in Colab
print("Please upload your dataset file (CSV format).")
uploaded = files.upload()  # Upload prompt

# Step 2: Load dataset (update the filename accordingly)
file_name = list(uploaded.keys())[0]  # Get the uploaded filename
df = pd.read_csv(file_name)

# Step 3: Define different question templates
question_templates = [
    "Can you summarize the key findings of this research paper?",
    "What problem does this research paper aim to solve?",
    "What are the main contributions of this study?",
    "How does this research advance the current state of knowledge?",
    "What methodology was used in this research?",
    "What are the potential applications of this research?",
    "How does this paper compare to previous work in this area?",
    "What challenges did the authors address in this paper?",
]

# Step 4: Create 'instruction', 'input', and 'output' columns
df["instruction"] = df.apply(lambda row: random.choice(question_templates), axis=1)

df["input"] = df.apply(
    lambda row: f"Title: {row['Title']}\nPublication Date: {row['publication_date']}\nArXiv Link: {row['arXiv_link']}",
    axis=1
)

df.rename(columns={"abstract": "output"}, inplace=True)  # Ensure abstract is renamed to output

# Step 5: Convert dataset to Unsloth's format
dataset = to_sharegpt(
    df,
    merged_prompt="instruction",  # Uses structured instruction
    input_column_name="input",  # Adds relevant input context
    output_column_name="output",  # Uses structured abstract as output
    conversation_extension=3,  # Optional for multi-turn conversations
)

# Step 6: Save and download the formatted dataset
formatted_file_name = "formatted_dataset.csv"
df.to_csv(formatted_file_name, index=False)
files.download(formatted_file_name)  # Auto-download

print(f"Dataset formatted successfully! 🚀 File saved as {formatted_file_name}")
