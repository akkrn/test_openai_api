import os

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")


# Define a function to analyze the sentiment and enthusiasm of a review using OpenAI API
def analyze_review(review):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Analyze the following review and rate its enthusiasm on a "
        f"scale of 1-10:\n\n{review}\n\nRating:",
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.7,
    )
    rating = int(response.choices[0].text)
    return rating


# Read the input table from a CSV file
input_filename = "input_table.csv"
df = pd.read_csv(input_filename)

# Analyze the sentiment and enthusiasm of each review and add the results to the table
df["rating"] = df["review"].apply(analyze_review)

# Sort the table in descending order of rating
df_sorted = df.sort_values(by=["rating"], ascending=False)

# Write the sorted table to a new CSV file
output_filename = f"{input_filename.split('.')[0]}_analyzed.csv"
df_sorted.to_csv(output_filename, index=False)

# Print the sorted table
print(df_sorted)
