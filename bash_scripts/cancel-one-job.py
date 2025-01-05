import os
from openai import OpenAI


openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

client = OpenAI(api_key=openai_api_key)

def cancel_job_by_id():
    try:
        # Prompt the user to enter the job ID
        batch_id = input("Enter the Batch ID to cancel: ").strip()

        job = client.batches.retrieve(batch_id)
        print(f"Job found: {job.id}, Status: {job.status}")
        if job.status == "in_progress":
            try:
                client.batches.cancel(batch_id)
                print(f"Successfully canceled batch job: {batch_id}")
            except Exception as e:
                print(f"Failed to cancel batch job {batch_id}: {e}")
        else:
            print(f"Batch job {batch_id} is not in progress (current status: {job.status}).")
    except Exception as e:
        print(f"Error while retrieving or canceling batch job: {e}")


cancel_job_by_id()

