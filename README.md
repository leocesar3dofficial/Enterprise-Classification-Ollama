# Enterprise Classification Ollama

A Python-based enterprise text classification system using Ollama models for efficient categorization of support tickets, feedback, and inquiries.

## Overview

This project provides tools for batch classification of text data using Ollama models, with support for analysis and benchmarking of classification results.

## Features

- Batch text classification using Ollama models
- Support for custom classification categories
- Results analysis with confusion matrix
- Performance benchmarking capabilities
- CSV input/output support

## Requirements

- Python 3.12+
- Ollama server running locally
- Required Python packages (install via `pip install -r requirements.txt`)

## Usage

1. Start your Ollama server and ensure it's accessible at the default endpoint:
```bash
export OLLAMA_ENDPOINT="http://localhost:11434/api/generate"
export MODEL="granite4"  # or your preferred model
```

2. Run batch classification:
```bash
python3 ollama_batch_classify.py --input benchmark_data.csv --output results.csv
```

3. Analyze results:
```bash
python3 analyze_results.py --input results.csv
```

## Classification Categories

The system classifies text into the following categories:
- Technical Support: bugs, errors, crashes, technical problems, system issues
- Billing: payments, invoices, refunds, subscriptions, charges, pricing
- Product Feedback: feature requests, suggestions, reviews, complaints, praise
- Account Management: passwords, login, user access, account settings, profile changes
- General Inquiry: general questions, policy questions, company info, availability

## Project Structure

- `ollama_batch_classify.py`: Main classification script
- `analyze_results.py`: Results analysis tool
- `generate_benchmark.py`: Benchmark data generation
- `requirements.txt`: Python dependencies
- `benchmark_data.csv`: Sample/benchmark dataset
- `results.csv`: Classification results output

## Contributing

Feel free to submit issues and enhancement requests!