#!/bin/bash

API_KEY="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"
curl -s -X GET "https://api.openai.com/v1/batches" \
-H "Authorization: Bearer $API_KEY" \
-H "Content-Type: application/json" | jq '.data[] | select(.status=="in_progress") | {id, created_at, request_counts}'
