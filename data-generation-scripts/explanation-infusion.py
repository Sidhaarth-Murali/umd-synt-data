import os
import json
import time
from openai import OpenAI
from huggingface_hub import HfApi
from datasets import load_dataset
from collections import defaultdict

# Load OpenAI and Hugging Face API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

client = OpenAI(api_key=openai_api_key)
hf_api = HfApi()
hf_token = "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"

# Define dataset-specific prompt templates
templates = {
    "math": {
        "prompt": lambda instruction: {
            "role": "user",
            "content": (
                "As an expert reasoning assistant, generate the following details based on the provided question and generate detailed explanation in a step-by-step manner to guide your responses.\n\n"
                f"Input: \"{instruction}\"\n\n"
                "Output JSON:\n{\n"
                "    \"response\": \"<model's detailed reasoning and answer>\",\n"
                "    \"conversations\": [\"<human query>\", \"<model's response>\"],\n"
                "    \"intent\": \"<intent of the question>\",\n"
                "    \"knowledge\": \"<knowledge used to solve this problem>\",\n"
                "    \"difficulty\": \"<difficulty level: easy, medium, hard>\",\n"
                "    \"task_category\": \"Math\",\n"
                "    \"input_quality\": \"<quality of the input>\"\n"
                "}\n"
            ),
        },
        "output_keys": [
            "response", "conversations", "intent", "knowledge",
            "difficulty", "task_category", "input_quality"
        ],
    },
    "code": {
        "prompt": lambda instruction: {
            "role": "user",
            "content": (
                "As an expert coding assistant, generate the following details based on the provided question and generate detailed explanation in a step-by-step manner to guide your responses..\n\n"
                f"Input: \"{instruction}\"\n\n"
                "Output JSON:\n{\n"
                "    \"response\": \"<model's solution and reasoning>\",\n"
                "    \"conversations\": [\"<human query>\", \"<model's response>\"],\n"
                "    \"intent\": \"<intent of the coding task>\",\n"
                "    \"knowledge\": \"<knowledge used to solve this task>\",\n"
                "    \"difficulty\": \"<difficulty level: easy, medium, hard>\",\n"
                "    \"task_category\": \"Coding & Debugging\",\n"
                "    \"input_quality\": \"<quality of the input>\"\n"
                "}\n"
            ),
        },
        "output_keys": [
            "response", "conversations", "intent", "knowledge",
            "difficulty", "task_category", "input_quality"
        ],
    },
    "instruction_following": {
        "prompt": lambda instruction: {
            "role": "user",
            "content": (
                "As an expert assistant for instruction-following tasks, generate the following details based on the provided question and generate detailed explanation in a step-by-step manner to guide your responses..\n\n"
                f"Input: \"{instruction}\"\n\n"
                "Output JSON:\n{\n"
                "    \"response\": \"<model's response to the instruction>\",\n"
                "    \"conversations\": [\"<human query>\", \"<model's response>\"],\n"
                "    \"intent\": \"<intent of the instruction>\",\n"
                "    \"knowledge\": \"<knowledge used to generate the response>\",\n"
                "    \"difficulty\": \"<difficulty level: easy, medium, hard>\",\n"
                "    \"task_category\": \"Instruction Following\",\n"
                "    \"input_quality\": \"<quality of the input>\"\n"
                "}\n"
            ),
        },
        "output_keys": [
            "response", "conversations", "intent", "knowledge",
            "difficulty", "task_category", "input_quality"
        ],
    },
}

# Load datasets
streaming_datasets = {
    "math": load_dataset("Magpie-Align/Magpie-Reasoning-150K", split="train", streaming=True),
    "code": load_dataset("Magpie-Align/Magpie-Reasoning-150K", split="train", streaming=True),
    "instruction_following": load_dataset("Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1", split="train", streaming=True),
}

# Filter datasets and limit to 10,000 examples
def filter_and_limit(dataset, filter_function, limit=10000):
    return dataset.filter(filter_function).take(limit)

datasets = {
    "math": filter_and_limit(streaming_datasets["math"], lambda x: x["task_category"] == "Math"),
    "code": filter_and_limit(streaming_datasets["code"], lambda x: x["task_category"] == "Coding & Debugging"),
    "instruction_following": streaming_datasets["instruction_following"].take(10000),
}

# # Generate batch tasks
def create_batch_tasks(dataset_name, dataset, template):
    tasks = []
    for idx, datapoint in enumerate(dataset):
        # Check if "instruction" exists
        instruction = datapoint.get("instruction", None)
        if not instruction:
            print(f"Skipping datapoint {idx} in {dataset_name}: 'instruction' is missing.")
            continue

        # Generate prompt
        dynamic_prompt = template["prompt"](instruction)

        # Create batch task
        task = {
            "custom_id": f"{dataset_name}-task-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": "You are an expert assistant."}, dynamic_prompt],
            },
        }
        tasks.append(task)
    return tasks

# Prepare and save batch tasks
batch_file_name = "batch_tasks_ei_mini.jsonl"
with open(batch_file_name, "w") as file:
    for dataset_name, dataset in datasets.items():
        template = templates[dataset_name]
        tasks = create_batch_tasks(dataset_name, dataset, template)
        for task in tasks:
            file.write(json.dumps(task) + "\n")

# Upload batch file
batch_file = client.files.create(file=open(batch_file_name, "rb"), purpose="batch")
print("Batch file uploaded:", batch_file.id)

# Create batch job
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
print("Batch job created:", batch_job.id)

# Monitor batch job
while True:
    batch_job_status = client.batches.retrieve("batch_675e73b1e58081909427ba32e3834db4")
    print(f"Batch job status: {batch_job_status.status}")
    if batch_job_status.status == "completed":
        break
    elif batch_job_status.status == "failed":
        raise RuntimeError("Batch job failed!")
    time.sleep(60)

# Retrieve results
result_file_id = batch_job_status.output_file_id
result_content = client.files.content(result_file_id).content

result_file_name = "batch_job_results_ei_mini.jsonl"
with open(result_file_name, "wb") as file:
    file.write(result_content)

# Save categorized results
def save_results_by_dataset(result_file_name):
    results = []
    with open(result_file_name, "r") as file:
        for line in file:
            result = json.loads(line.strip())
            results.append(result)

    categorized_results = {"math": [], "code": [], "instruction_following": []}

    for res in results:
        custom_id = res["custom_id"]
        dataset_name = custom_id.split("-")[0]
        categorized_results[dataset_name].append(res["response"]["body"]["choices"][0]["message"]["content"])

    for dataset_name, data in categorized_results.items():
        output_file = f"synthetic_{dataset_name}_ei_mini.jsonl"
        with open(output_file, "w") as file:
            for item in data:
                file.write(json.dumps(item) + "\n")
        print(f"Results saved to {output_file}")

save_results_by_dataset(result_file_name)

# Upload results to Hugging Face
for dataset_name in templates.keys():
    output_file = f"synthetic_{dataset_name}_ei_mini.jsonl"
    hf_repo_name = f"SidhaarthMurali/synthetic_{dataset_name}_ei_mini"
    hf_api.create_repo(repo_id=hf_repo_name, repo_type="dataset", token=hf_token, exist_ok=True)
    hf_api.upload_file(
        path_or_fileobj=output_file,
        path_in_repo=f"{dataset_name}_ei_mini.jsonl",
        repo_id=hf_repo_name,
        repo_type="dataset",
        token=hf_token
    )
    print(f"Uploaded {dataset_name} to Hugging Face as a dataset: {hf_repo_name}")
