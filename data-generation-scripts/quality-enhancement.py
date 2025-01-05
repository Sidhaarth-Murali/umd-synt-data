import os
import json
import time
from openai import OpenAI
from huggingface_hub import HfApi
from datasets import load_dataset

# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# client = OpenAI(api_key=openai_api_key)
hf_api = HfApi()
hf_token = "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"

# Define dataset-specific prompt templatess
templates = {
    "math": {
        "prompt": lambda datapoint: (
            "As an expert reasoning assistant, enhance the quality of the response by thinking deeply about the problem.\n\n"
            f"Input: \"{datapoint['model_output']}\"\n\n"
            f"Problem Statement: \"{datapoint['problem_statement']}\"\n\n"
            f"Golden Answer: \"{datapoint['golden_answer']}\"\n\n"
            "Output JSON:\n{\n"
            "    \"enhanced_response\": \"<enhanced model output>\",\n"
            "    \"justification\": \"<justification for enhancements>\",\n"
            "    \"uuid\": \"{datapoint['uuid']}\",\n"
            "    \"task_category\": \"Math\"\n"
            "}"
        ),
        "output_keys": ["enhanced_response", "justification", "uuid", "task_category"]
    },
    "code": {
        "prompt": lambda datapoint: (
            "As an expert coding assistant, enhance the quality of the response by thinking deeply about the rewritten intent and the provided snippet.\n\n"
            f"Input: \"{datapoint['rewritten_intent']}\"\n\n"
            f"Snippet: \"{datapoint['snippet']}\"\n\n"
            "Output JSON:\n{\n"
            "    \"enhanced_snippet\": \"<enhanced code snippet>\",\n"
            "    \"justification\": \"<justification for enhancements>\",\n"
            "    \"question_id\": \"{datapoint['question_id']}\",\n"
            "    \"task_category\": \"Coding\"\n"
            "}"
        ),
        "output_keys": ["enhanced_snippet", "justification", "question_id", "task_category"]
    },
    "instruction_following": {
        "prompt": lambda datapoint: (
            "As an expert assistant, enhance the quality of the response by thinking deeply about the instruction and context.\n\n"
            f"Instruction: \"{datapoint['instruction']}\"\n\n"
            f"Context: \"{datapoint['context']}\"\n\n"
            f"Response: \"{datapoint['response']}\"\n\n"
            "Output JSON:\n{\n"
            "    \"enhanced_response\": \"<enhanced model output>\",\n"
            "    \"justification\": \"<justification for enhancements>\",\n"
            "    \"category\": \"{datapoint['category']}\",\n"
            "    \"task_category\": \"Instruction Following\"\n"
            "}"
        ),
        "output_keys": ["enhanced_response", "justification", "category", "task_category"]
    }
}


# # Load datasets
# def filter_and_limit(dataset, filter_function=None, limit=10000):
#     if filter_function:
#         dataset = dataset.filter(filter_function)
#     return dataset.select(range(min(len(dataset), limit)))

# datasets = {
#     "math": filter_and_limit(
#         load_dataset("toloka/mu-math", split="test"),
#         None,  # No filtering
#         limit=10000
#     ),
#     "code": filter_and_limit(
#         load_dataset("neulab/conala", "curated", split="train"),
#         None,  # No filtering
#         limit=10000
#     ),
#     "instruction_following": filter_and_limit(
#         load_dataset("databricks/databricks-dolly-15k", split="train"),
#         lambda x: x["category"] in ["open_qa", "general_qa", "brainstorming"],
#         limit=10000
#     )
# }

# # Generate batch tasks
# def create_batch_tasks(dataset_name, dataset, template):
#     tasks = []
#     for idx, datapoint in enumerate(dataset):
#         if dataset_name == "math":
#             dynamic_prompt = template["prompt"](datapoint)
#         elif dataset_name == "code":
#             dynamic_prompt = template["prompt"](datapoint)

#         elif dataset_name == "instruction_following":
#             dynamic_prompt = template["prompt"](datapoint)
#         else:
#             continue

#         task = {
#             "custom_id": f"{dataset_name}-task-{idx}_mini",
#             "method": "POST",
#             "url": "/v1/chat/completions",
#             "body": {
#                 "model": "gpt-4o-mini",
#                 "temperature": 0.7,
#                 "response_format": {"type": "json_object"},
#                 "messages": [
#                     {"role": "system", "content": "You are an expert assistant."},
#                     {"role": "user", "content": dynamic_prompt}
#                 ]
#             }
#         }
#         tasks.append(task)
#     return tasks

# # Prepare and save batch tasks
# batch_file_name = "batch_tasks_qe_mini.jsonl"
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

# Monitor batch job
# while True:
#     batch_job_status = client.batches.retrieve("batch_675e714db9dc8190b50b8d5e548b653d")
#     print(f"Batch job status: {batch_job_status.status}")
#     if batch_job_status.status == "completed":
#         break
#     elif batch_job_status.status == "failed":
#         raise RuntimeError("Batch job failed!")
#     time.sleep(60)

# # Retrieve results
# result_file_id = batch_job_status.output_file_id
# result_content = client.files.content(result_file_id).content

# result_file_name = "batch_job_results_qe_mini.jsonl"
# with open(result_file_name, "wb") as file:
#     file.write(result_content)

# # Save categorized results
# def save_results_by_dataset(result_file_name):
#     results = []
#     with open(result_file_name, "r") as file:
#         for line in file:
#             result = json.loads(line.strip())
#             results.append(result)

#     categorized_results = {"math": [], "code": [], "instruction_following": []}

#     for res in results:
#         custom_id = res["custom_id"]
#         dataset_name = custom_id.split("-")[0]
#         categorized_results[dataset_name].append(res["response"]["body"]["choices"][0]["message"]["content"])

#     for dataset_name, data in categorized_results.items():
#         output_file = f"synthetic_{dataset_name}_qe_mini.jsonl"
#         with open(output_file, "w") as file:
#             for item in data:
#                 file.write(json.dumps(item) + "\n")
#         print(f"Results saved to {output_file}")

# save_results_by_dataset(result_file_name)

# Upload results to Hugging Face
for dataset_name in templates.keys():
    dataset_name = "instruction_following"
    output_file = f"synthetic_{dataset_name}_qe.jsonl"
    hf_repo_name = f"SidhaarthMurali/synthetic_{dataset_name}_qe_"
    hf_api.create_repo(repo_id=hf_repo_name, repo_type="dataset", token=hf_token, exist_ok=True)
    hf_api.upload_file(
        path_or_fileobj=output_file,
        path_in_repo=f"/{dataset_name}_qe.jsonl",
        repo_id=hf_repo_name,
        repo_type="dataset",
        token=hf_token
    )
    print(f"Uploaded {dataset_name} to Hugging Face as a dataset: {hf_repo_name}")
    break
