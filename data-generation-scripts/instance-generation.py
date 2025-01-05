import os
from openai import OpenAI
from huggingface_hub import HfApi, HfFolder
from datasets import load_dataset
import json
import asyncio
import time

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

client = OpenAI(api_key=openai_api_key)
hf_api = HfApi()
hf_token = "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"

# Define the dataset templates and generation prompts
templates = {
    "gsm8k": {
        "prompt": "As a data generator, your task is to generate a new example (`question` and `answer`) for a dataset similar to GSM8K and MATH. Each example should be challenging yet solvable and formatted in JSON. Please provide the following format:\n\n{\n    \"question\": \"<input>\",\n    \"answer\": \"<output>\"\n}\n\nExample 1:",
        "output_key": "question"
    },
    "mbpp": {
        "prompt": "As a data generator, your task is to generate a new example (`task_id`, `text`, `code`, `test_list`, `test_setup_code`, `challenge_test_list`) for a dataset similar to MBPP and xP3x. Each example should be challenging yet solvable and formatted in JSON. Please provide the following format:\n\n{\n    \"task_id\": <integer>,\n    \"text\": \"<input>\",\n    \"code\": \"<output>\",\n    \"test_list\": [<list_of_tests>],\n    \"test_setup_code\": \"<setup_code>\",\n    \"challenge_test_list\": [<list_of_challenges>]\n}\n\nExample 1:",
        "output_key": "text"
    },
    "lima": {
        "prompt": "As a data generator, your task is to generate a new example (`conversations` and `source`) for a dataset similar to LIMA. Each example should be mindful, original, factual and formatted in JSON. Please provide the following format:\n\n{\n    \"conversations\": [\"<input>\", \"<output>\"],\n    \"source\": \"<source_information>\"\n}\n\nExample 1:",
        "output_key": "conversations"
    }
}

# Load datasets and determine their lengths
datasets = {
    "gsm8k": load_dataset("openai/gsm8k", "main", split="train"),
    "mbpp": load_dataset("google-research-datasets/mbpp", split="train"),
    "lima": load_dataset("GAIR/lima", split="train")
}

data_counts = {name: len(dataset) for name, dataset in datasets.items()}

# Define models
models = ["gpt-4o-mini"]

# # Prepare batch tasks with dynamic dataset prompts
# def create_batch_tasks(dataset_name, dataset, template):
#     tasks = []
#     for idx, datapoint in enumerate(dataset):
#         example = json.dumps(datapoint, indent=4)
#         dynamic_prompt = f"{template['prompt']}\n\n{example}\n\nNew Example:"

#         for model in models:
#             task = {
#                 "custom_id": f"{dataset_name}-{model}-task-{idx}",
#                 "method": "POST",
#                 "url": "/v1/chat/completions",
#                 "body": {
#                     "model": model,
#                     "temperature": 0.1,
#                     "response_format": {"type": "json_object"},
#                     "messages": [
#                         {"role": "system", "content": "You are an expert data generator."},
#                         {"role": "user", "content": dynamic_prompt}
#                     ]
#                 }
#             }
#             tasks.append(task)
#     return tasks

# # Generate tasks for all datasets
# batch_file_name = "batch_tasks_ig_mini.jsonl"
# with open(batch_file_name, "w") as file:
#     for dataset_name, dataset in datasets.items():
#         template = templates[dataset_name]
#         tasks = create_batch_tasks(dataset_name, dataset, template)
#         for task in tasks:
#             file.write(json.dumps(task) + "\n")

# # Upload batch file
# batch_file = client.files.create(file=open(batch_file_name, "rb"), purpose="batch")
# print("Batch file uploaded:", batch_file.id)

# # Create batch job
# batch_job = client.batches.create(
#     input_file_id=batch_file.id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h"
# )
# print("Batch job created:", batch_job.id)

while True:
    batch_job_status = client.batches.retrieve("batch_675e70c9cb148190b31ca9c5e35337a2")
    print(f"Batch job status: {batch_job_status.status}")
    if batch_job_status.status == "completed":
        break
    elif batch_job_status.status == "failed":
        raise RuntimeError("Batch job failed!")
    time.sleep(60)

# Retrieve results
result_file_id = batch_job_status.output_file_id
result_content = client.files.content(result_file_id).content

result_file_name = "batch_job_results_ig_mini.jsonl"
with open(result_file_name, "wb") as file:
    file.write(result_content)

# Retrieve and categorize results
def save_results_by_dataset(result_file_name):
    results = []
    with open(result_file_name, "r") as file:
        for line in file:
            result = json.loads(line.strip())
            results.append(result)

    categorized_results = {"gsm8k": [], "mbpp": [], "lima": []}

    for res in results:
        custom_id = res["custom_id"]
        dataset_name, model_name = custom_id.split("-")[:2]  # Extract dataset and model name
        categorized_results[dataset_name].append({"model": model_name, "content": res["response"]["body"]["choices"][0]["message"]["content"]})

    for dataset_name, data in categorized_results.items():
        output_file = f"synthetic_{dataset_name}_ig_mini.jsonl"
        with open(output_file, "w") as file:
            for item in data:
                file.write(json.dumps(item) + "\n")
        print(f"Results saved to {output_file}")

    return categorized_results

# Save categorized results to Hugging Face
categorized_results = save_results_by_dataset(result_file_name)

for dataset_name in data_counts.keys():
    # Define the output file path
    output_file = f"synthetic_{dataset_name}_ig_mini.jsonl"
    
    # Define the dataset repository name
    hf_repo_name = f"SidhaarthMurali/synthetic_{dataset_name}_ig_mini"
    
    # Create the dataset repository if it doesn't exist
    try:
        hf_api.create_repo(repo_id=hf_repo_name, repo_type="dataset", token=hf_token, exist_ok=True)
        print(f"Dataset repository created (or already exists): {hf_repo_name}")
    except Exception as e:
        print(f"Error creating dataset repository: {e}")
    
    # Upload the dataset file to the dataset repository
    try:
        hf_api.upload_file(
            path_or_fileobj=output_file,
            path_in_repo=f"{dataset_name}_ig_mini.jsonl",  # Specify a clean file name
            repo_id=hf_repo_name,
            repo_type="dataset",  # Specify the repository type as "dataset"
            token=hf_token
        )
        print(f"Uploaded {dataset_name} to Hugging Face as a dataset: {hf_repo_name}")
    except Exception as e:
        print(f"Error uploading file to dataset repository: {e}")

