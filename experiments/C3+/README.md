# C3+ Framework

## Current Status

Combined modules provided in C3 and DIN to create C3+.

Slight enhancements made to the "classification" step using confidence scores and fallbacks.

The code works and it ran for 1 instance of the queries.

No further instances have been run due to cost, maybe will run until the pipeline has had significant changes made to it.

## What's Next

Going to try and modify the Schema Linking by introducing an automated filtering step that uses a similarity metric (cosine similarity over embeddings) to determine the most relevant schema elements.

Grounds: https://arxiv.org/html/2403.16204v1

An easy fix to some common errors: integrate an intermediate “SQL validator” function that checks for syntax errors or common logical mistakes before passing the query to the self-correction module.
