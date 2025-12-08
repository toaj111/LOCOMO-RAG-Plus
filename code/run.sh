#!/bin/bash
uv run run_experiments.py --technique_type mem0 --method add --output_folder results/ --top_k 10
uv run run_experiments.py --technique_type mem0 --method search --output_folder results/ --top_k 10  
uv run evals.py --input_file results/mem0_results_top_10_filter_False_graph_False.json --max_workers 50
cp -r results ../results
