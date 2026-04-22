from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # Can increase for longer reasoning traces
lora_rank = 64  # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path_to_model",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.9,  # Reduce if out of memory
)

import re

# Regex to match <think>...</think> followed by <diagnosis>...</diagnosis>
think_end_regex = r"</think>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"<think>.*?</think>[\s]{{0,}}<diagnosis>(.+?){think_end_regex}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
)


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many tags are present - partial credit
        score += 0.5 if response.count("<think>") == 1 else -1.0
        score += 0.5 if response.count("</think>") == 1 else -1.0
        score += 0.5 if response.count("<diagnosis>") == 1 else -1.0
        score += 0.5 if response.count("</diagnosis>") == 1 else -1.0
        scores.append(score)
    return scores


def match_diagnosis_answer(completions, **kwargs):
    true_answer = kwargs.get("diagnosis", "")
    if not true_answer:
        return [0.0] * len(completions)
    scores = []
    for completion in completions:
        score = 0.0
        response = completion[0]["content"]
        diagnosis_match = re.search(
            rf"<diagnosis>(.+?)</diagnosis>",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if diagnosis_match:
            diagnosis_text = diagnosis_match.group(1).lower()
            if true_answer.lower() in diagnosis_text:
                score += 10.0
        scores.append(score)
    return scores


# --- Judge LLM for evaluation ---
from unsloth import FastLanguageModel as UnslothModel

judge_model, judge_tokenizer = UnslothModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
    gpu_memory_utilization=0.5,
)
judge_model = UnslothModel.get_peft_model(
    judge_model, r=0, use_gradient_checkpointing="unsloth", random_state=3407
)

JUDGE_PROMPT = """\
You are a medical evaluation expert. Evaluate the following model response to a clinical case.
Rate it on a scale of 0 to 100 based on:
1. Clinical accuracy - is the diagnosis and reasoning medically correct?
2. Reasoning quality - is the thought process logical, thorough, and well-structured?

Output ONLY a single integer score between 0 and 100. Do not output anything else.

Model response:
{response}
"""


def judge_reward_fn(completions, prompt, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        judge_input = JUDGE_PROMPT.format(response=response)
        messages = [{"role": "user", "content": judge_input}]
        text = judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = judge_model.generate(
            inputs=judge_tokenizer(text, return_tensors="pt").to(judge_model.device),
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
        )
        generated_text = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the number from the output
        try:
            score_100 = int(generated_text.strip().split()[-1])
            score_100 = max(0, min(100, score_100))
        except (ValueError, IndexError):
            score_100 = 50  # fallback if parsing fails
        scores.append(score_100 / 20.0)  # scale 0-100 to 0-5
    return scores


max_prompt_length = "after dataset"  # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

from vllm import SamplingParams

vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Increase to 4 for smoother training
    num_generations=4,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps=100,
    save_steps=100,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

# --- Combined reward function for GRPOTrainer ---
# Each reward function receives (completions, prompt, **kwargs) and returns a list of scores
# GRPOTrainer expects reward_f to take (completions, prompt, **kwargs) -> list[float]

def combined_reward_fn(completions, prompt, **kwargs):
    total_scores = []
    for i in range(len(completions)):
        total_scores.append(0.0)

    # Format rewards
    format_scores = match_format_exactly(completions, **kwargs)
    for i in range(len(completions)):
        total_scores[i] += format_scores[i]

    format_scores = match_format_approximately(completions, **kwargs)
    for i in range(len(completions)):
        total_scores[i] += format_scores[i]

    # Diagnosis keyword match
    diag_scores = match_diagnosis_answer(completions, **kwargs)
    for i in range(len(completions)):
        total_scores[i] += diag_scores[i]

    # Judge LLM evaluation
    judge_scores = judge_reward_fn(completions, prompt, **kwargs)
    for i in range(len(completions)):
        total_scores[i] += judge_scores[i]

    return total_scores