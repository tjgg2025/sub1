# humaneval

python main.py \
    --run_name "simple_llama3_pass1_humanEval" \
    --root_dir ./results/simple_try/humanEval/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "simple" \
    --language "py" \
    --model "qwen3_70b" \
    --pass_at_k "1" \
    --max_iters "1" \
    --verbose


python main.py \
  --run_name "reflexion_llama3_8B_pass1_humanEvalFull" \
  --root_dir   ./results/reflexion_Final/humanEval/ \
  --dataset_path benchmarks/humaneval_full.jsonl \
  --strategy   "reflexion" \
  --language   "py" \
  --model      "llama3_1_8b" \
  --pass_at_k  "1" \
  --max_iters  "5" \
  --verbose


python main.py \
    --run_name "dot_llama3_8B_pass1_humanEval" \
    --root_dir ./results/DoT_Final/humanEval/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "dot" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --verbose

python main.py \
    --run_name "dot_bank_llama3_8B_pass1_humanEval" \
    --root_dir ./results/DoT_Final/humanEval/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "dot_bank" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --verbose


python main_param.py \
    --run_name "simple_llama3_pass1_humanEval_model_based_reflexion" \
    --root_dir ./results/llama_rerun/simple_Final/humanEval/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "simple" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --use_mistakes \
    --device cuda:1 \
    --mistake_json_path ./benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl \
    --verbose


python main_param.py \
    --run_name "dot_parametric_llama3_8B_pass1_humanEval" \
    --root_dir ./results/icml_result/DoT_Final/humanEval/ParamAgent/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "dot" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --inner_iter "5" \
    --use_mistakes \
    --mistake_json_path ./benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl \
    --verbose

# mbpp


python main.py \
    --run_name "simple_llama3_pass1_mbpp_raw_try" \
    --root_dir ./results/simple_Final/mbpp/ \
    --dataset_path benchmarks/mbpp-py.jsonl \
    --strategy "simple" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "1" \
    --verbose


python main_param.py \
    --run_name "simple_llama3_pass1_mbpp_model_based_reflexion_try" \
    --root_dir ./results/llama_rerun/simple_Final/mbpp/ \
    --dataset_path benchmarks/mbpp-py.jsonl \
    --strategy "simple" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --use_mistakes \
    --device cuda:1 \
    --mistake_json_path ./benchmarks/code_pitfalls/mbpp_pitfalls.jsonl \
    --verbose

# python main.py \
#   --run_name "reflexion_llama3_8B_pass1_mbpp" \
#   --root_dir   ./results/reflexion_Final/mbpp/ \
#   --dataset_path benchmarks/mbpp-py.jsonl \
#   --strategy   "reflexion" \
#   --language   "py" \
#   --model      "llama3_1_8b" \
#   --pass_at_k  "1" \
#   --max_iters  "5" \
#   --verbose


# python main.py \
#     --run_name "dot_llama3_8B_pass1_mbpp" \
#     --root_dir ./results/DoT_Final/mbpp/ \
#     --dataset_path benchmarks/mbpp-py.jsonl \
#     --strategy "dot" \
#     --language "py" \
#     --model "llama3_1_8b" \
#     --pass_at_k "1" \
#     --max_iters "5" \
#     --verbose


python main.py \
    --run_name "dot_bank_llama3_8B_pass1_mbpp" \
    --root_dir ./results/DoT_Final/mbpp/ \
    --dataset_path benchmarks/mbpp-py.jsonl \
    --strategy "dot_bank" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --verbose


python main.py \
    --run_name "dot_bank_llama3_8B_pass1_mbpp" \
    --root_dir ./results/DoT_Bank/mbpp/ \
    --dataset_path benchmarks/mbpp-py.jsonl \
    --strategy "dot_bank" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "3" \
    --verbose

python main.py \
    --run_name "dot_bank_qwen_pass1_mbpp" \
    --root_dir ./results/DoT_Bank/mbpp/ \
    --dataset_path benchmarks/mbpp-py.jsonl \
    --strategy "dot_bank" \
    --language "py" \
    --model "qwen_1.5b" \
    --pass_at_k "1" \
    --max_iters "4" \
    --verbose



python main_param.py \
    --run_name "dot_parametric_llama3_8B_pass1_mbpp" \
    --root_dir ./results/DoT_Final/mbpp/ \
    --dataset_path benchmarks/mbpp-py.jsonl \
    --strategy "dot" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --inner_iter "5" \
    --use_mistakes \
    --mistake_json_path ./benchmarks/code_pitfalls/mbpp_pitfalls.jsonl \
    --verbose


