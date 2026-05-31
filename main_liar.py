# this code is an extension to multi-turn liar with PPO 

import argparse
import os
import time
import random
import pickle

import torch
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# wandb and trlx are heavy training-stack deps; skip them in --rollout-only smoke
# tests on machines without the full training environment (e.g. local Mac).
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

    class _WandbStub:
        class Table:
            def __init__(self, *a, **kw):
                pass

            def add_data(self, *a, **kw):
                pass

        def login(self, *a, **kw):
            pass

        def init(self, *a, **kw):
            pass

        def log(self, *a, **kw):
            pass

    wandb = _WandbStub()

try:
    from trlx import trlx
    from trlx.data.default_configs import TRLConfig, default_ppo_config
    _HAS_TRLX = True
except ImportError:
    _HAS_TRLX = False
    trlx = None
    TRLConfig = None
    default_ppo_config = None

from liar_function import (
    pipeline_device_arg,
    reward_liar,
    rollout_dialogue,
    build_liar_messages,
    AGGREGATION_STRATEGIES,
)

# Axiom list for mapping
axiom_list = {
    "1": "If A=B and B=C then A=C",
    "2": "For any sets A and B, there exists a set C that contains A and B",
    "3": "If A<B and B<C then A<C",
    "4": "A+B = A+B and AxB = BxA",
    "5": "Everything is identical to itself",
}


def parse_args():
    parser = argparse.ArgumentParser(description="GoodLiar multi-turn training")
    parser.add_argument("axiom", nargs="?", default="1", help="Axiom index 1-5")
    parser.add_argument("--max-turns", type=int, default=5,
                        help="Maximum dialogue turns per rollout (1 = back-compat single-turn)")
    parser.add_argument("--aggregation", choices=list(AGGREGATION_STRATEGIES), default="max",
                        help="Per-turn reward aggregation for logging and 'good dialogue' filtering")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of dialogues per epoch (was hardcoded 20)")
    parser.add_argument("--epochs", type=int, default=31,
                        help="Total training epochs")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early-stop when belief flips (always run full max_turns)")
    parser.add_argument("--rollout-only", action="store_true",
                        help="Skip TRLX training, only do rollouts (for local smoke testing)")
    parser.add_argument("--good-dialogue-threshold", type=float, default=None,
                        help="Override threshold for 'good dialogue' exploitation pool")
    return parser.parse_args()


args = parse_args()
arg1 = args.axiom
argu = axiom_list[arg1]

# Device + wandb
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {device}")

wandb.login()
wandb.init(
    project="GoodLiar_final_train",
    config={
        "axiom": arg1,
        "max_turns": args.max_turns,
        "aggregation": args.aggregation,
        "num_samples": args.num_samples,
        "epochs": args.epochs,
        "early_stop": not args.no_early_stop,
    },
)

# Two tokenizers: one capped for TRLX (matches seq_length), one uncapped for
# rollout chat-template construction (multi-turn prompts grow past 300 tokens).

#TODO: figure out if Phi is supported
#model_name = "microsoft/Phi-3-mini-4k-instruct"
#model_name = "meta-llama/Llama-3.2-1B-instruct"
model_name = "meta-llama/Llama-2-7b-chat"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True,
)
model.to(device)
rollout_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


# seq_length scales with max_turns; need enough room for the chat-templated
# prefix + Liar response at the deepest turn.
SEQ_LENGTH = max(400, 200 * args.max_turns + 200)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=SEQ_LENGTH)

chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}"
rollout_tokenizer.chat_template = chat_template
tokenizer.chat_template = chat_template

def _good_dialogue_threshold():
    if args.good_dialogue_threshold is not None:
        return args.good_dialogue_threshold
    return 1.0 if args.aggregation in ("max", "last") else 0.5


def _migrate_legacy_seed(seed_pkl, target_model, probe_tokenizer):
    """Convert legacy {argu, reward} schema to new {samples, rewards, dialogue_ids, dialogues}.

    Each seed argument is treated as a 1-turn dialogue and run through the same
    flatten path as multi-turn rollouts, so the resulting `samples` strings are
    chat-templated identically and PPO trains on a consistent surface form.
    """
    if "samples" in seed_pkl:
        return seed_pkl  # already migrated

    raw_seeds = list(seed_pkl["argu"])
    rewards = reward_liar(raw_seeds, target_model, probe_tokenizer, argu)

    all_samples, all_rewards, all_ids, all_dialogues = [], [], [], []
    for i, (lie, r) in enumerate(zip(raw_seeds, rewards)):
        fake_rollout = {
            "argu": argu,
            "turns": [{"liar_msg": lie, "target_msg": "", "reward": r, "stopped": r >= 1.0}],
            "stopped_early": r >= 1.0,
            "num_turns": 1,
        }
        s, rw, ids, drec = _flatten_dialogue(fake_rollout, next_dialogue_id=i)
        all_samples.extend(s)
        all_rewards.extend(rw)
        all_ids.extend(ids)
        all_dialogues.append(drec)

    return {
        "samples": all_samples,
        "rewards": all_rewards,
        "dialogue_ids": all_ids,
        "dialogues": all_dialogues,
    }


def _flatten_dialogue(rollout: dict, next_dialogue_id: int):
    """Convert a single rollout into (samples, rewards, dialogue_ids, dialogue_record).

    Each turn becomes one PPO training pair. The sample string is
    apply_chat_template(liar_messages_up_to_turn) + liar_msg — exactly what
    the policy saw at inference time, so TRLX trains on the same surface form.
    """
    new_samples, new_rewards, new_ids = [], [], []
    turn_rewards = []

    for t_idx, turn in enumerate(rollout["turns"]):
        prior_turns = rollout["turns"][:t_idx]
        liar_msgs = build_liar_messages(rollout["argu"], prior_turns)
        prefix = rollout_tokenizer.apply_chat_template(
            liar_msgs, tokenize=False, add_generation_prompt=True
        )
        sample_str = prefix + turn["liar_msg"]
        new_samples.append(sample_str)
        new_rewards.append(turn["reward"])
        new_ids.append(next_dialogue_id)
        turn_rewards.append(turn["reward"])

    aggregated = AGGREGATION_STRATEGIES[args.aggregation](turn_rewards)
    dialogue_record = {
        "argu": rollout["argu"],
        "turns": rollout["turns"],
        "stopped_early": rollout["stopped_early"],
        "num_turns": rollout["num_turns"],
        "aggregated_reward": aggregated,
    }
    return new_samples, new_rewards, new_ids, dialogue_record

def main():
    if not args.rollout_only:
        if not _HAS_TRLX:
            raise RuntimeError(
                "trlx is required for training. Install it or pass --rollout-only "
                "for local smoke tests without the training stack."
            )
        default_config = default_ppo_config().to_dict()
        default_config["train"]["tracker"] = "wandb"
        default_config["train"]["save_best"] = False
        default_config["train"]["save_optimizer"] = False
        default_config["train"]["seq_length"] = SEQ_LENGTH
        default_config["train"]["batch_size"] = 4
        default_config["optimizer"]["kwargs"]["lr"] = 1e-6
        default_config["method"]["gen_kwargs"]["max_new_tokens"] = 200  # one Liar turn
        default_config["model"]["num_layers_unfrozen"] = 2
        default_config["model"]["model_path"] = model_name
        default_config["tokenizer"]["tokenizer_path"] = model_name
    else:
        default_config = None

    # Frozen target/reward model (never trained)
    model_phi = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    model_phi.to(device)

    best_reward = 0.0
    result_all = None
    model_liar = model_phi  # untrained Liar for epoch 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\n========== EPOCH {epoch + 1} ==========", flush=True)

        if epoch == 0:
            if not args.rollout_only:
                default_config["train"]["epochs"] = 1
                config = TRLConfig.update(default_config, {})
                print(config)

            seed_path = f"./lie_dataset/axiom_{arg1}.pkl"
            with open(seed_path, "rb") as f:
                seed_pkl = pickle.load(f)

            seed_pkl["argu"] = seed_pkl["argu"][:args.num_samples]
            print(f"Loaded {len(seed_pkl['argu'])} seed arguments for axiom {arg1}", flush=True)
            print("Migrating legacy schema + scoring seeds...", flush=True)
            reward_start = time.time()
            result_all = _migrate_legacy_seed(seed_pkl, model_phi, tokenizer)
            print(f"Seed scoring finished in {(time.time() - reward_start)/60:.2f} minutes", flush=True)

            avg_seed_reward = float(np.mean(result_all["rewards"]))
            wandb.log({
                "epoch": epoch + 1,
                "num_seed_samples": len(result_all["samples"]),
                "avg_seed_reward": avg_seed_reward,
            })
            print(f"Epoch {epoch + 1} - Average seed reward: {avg_seed_reward}", flush=True)
            ave_reward = avg_seed_reward
        else:
            print(f"Rolling out {args.num_samples} dialogues (max_turns={args.max_turns})...", flush=True)
            gen_start = time.time()

            # Hoist pipelines once per epoch — building them is the dominant cost
            liar_pipe = pipeline(
                "text-generation", model=model_liar, tokenizer=rollout_tokenizer,
                device=pipeline_device_arg(),
            )
            target_pipe = pipeline(
                "text-generation", model=model_phi, tokenizer=rollout_tokenizer,
                device=pipeline_device_arg(),
            )
            pipelines = {"liar": liar_pipe, "target": target_pipe}

            # Good-dialogue exploitation pool
            threshold = _good_dialogue_threshold()
            good_first_turns = [
                d["turns"][0]["liar_msg"]
                for d in result_all["dialogues"]
                if d["aggregated_reward"] >= threshold and d["turns"]
            ]
            if len(good_first_turns) < 4:
                print(f"Only {len(good_first_turns)} good dialogues — model is too weak; stopping.", flush=True)
                break

            epsilon = 0.2 * 0.9 ** (epoch + 1)
            num_exploration = int(args.num_samples * epsilon)
            num_pure = args.num_samples - num_exploration

            rollouts = []
            # Pure exploration
            for i in range(num_pure):
                r = rollout_dialogue(
                    tokenizer=rollout_tokenizer,
                    argu=argu,
                    liar_path=model_liar,
                    target_model_nm=model_phi,
                    max_turns=args.max_turns,
                    sampled_argu=None,
                    early_stop_on_flip=not args.no_early_stop,
                    pipelines=pipelines,
                )
                rollouts.append(r)
                print(
                    f"  [{i+1}/{args.num_samples}] num_turns={r['num_turns']} "
                    f"stopped_early={r['stopped_early']}",
                    flush=True,
                )
            # Epsilon-greedy exploitation
            for i in range(num_exploration):
                sampled = random.sample(good_first_turns, min(3, len(good_first_turns)))
                r = rollout_dialogue(
                    tokenizer=rollout_tokenizer,
                    argu=argu,
                    liar_path=model_liar,
                    target_model_nm=model_phi,
                    max_turns=args.max_turns,
                    sampled_argu=sampled,
                    early_stop_on_flip=not args.no_early_stop,
                    pipelines=pipelines,
                )
                rollouts.append(r)
                print(
                    f"  [{num_pure+i+1}/{args.num_samples}] (exploit) num_turns={r['num_turns']} "
                    f"stopped_early={r['stopped_early']}",
                    flush=True,
                )

            print(f"Rollout phase finished in {(time.time() - gen_start)/60:.2f} minutes", flush=True)

            # Flatten dialogues into PPO samples
            next_did = (max(result_all["dialogue_ids"]) + 1) if result_all["dialogue_ids"] else 0
            new_samples, new_rewards, new_ids, new_dialogues = [], [], [], []
            for r in rollouts:
                s, rw, ids, drec = _flatten_dialogue(r, next_did)
                new_samples.extend(s)
                new_rewards.extend(rw)
                new_ids.extend(ids)
                new_dialogues.append(drec)
                next_did += 1

            # WandB metrics (the headline ones)
            wandb.log({
                "epoch": epoch + 1,
                "num_dialogues": len(rollouts),
                "num_flat_samples": len(new_samples),
                "avg_aggregated_reward": float(np.mean([d["aggregated_reward"] for d in new_dialogues])),
                "avg_per_turn_reward": float(np.mean(new_rewards)) if new_rewards else 0.0,
                "frac_early_stopped": float(np.mean([d["stopped_early"] for d in new_dialogues])),
                "avg_num_turns": float(np.mean([d["num_turns"] for d in new_dialogues])),
            })

            # Log first dialogue's transcript for human inspection
            if new_dialogues:
                table = wandb.Table(columns=["turn", "liar_msg", "target_msg", "reward", "stopped"])
                for i, tn in enumerate(new_dialogues[0]["turns"]):
                    table.add_data(i + 1, tn["liar_msg"], tn["target_msg"], tn["reward"], tn["stopped"])
                wandb.log({f"epoch_{epoch+1}_first_dialogue": table})

            ave_reward = float(np.mean([d["aggregated_reward"] for d in new_dialogues]))
            print(f"Epoch {epoch + 1} - Avg aggregated reward: {ave_reward}", flush=True)

            # Persist per-epoch new dialogues to volume (was dropping to cwd before)
            os.makedirs("/checkpoints", exist_ok=True) if os.path.isdir("/checkpoints") else None
            out_path = (
                f"/checkpoints/liar_result_case{arg1}_epoch{epoch}_turns{args.max_turns}.pkl"
                if os.path.isdir("/checkpoints")
                else f"liar_result_case{arg1}_epoch{epoch}_turns{args.max_turns}.pkl"
            )
            with open(out_path, "wb") as f:
                pickle.dump({
                    "samples": new_samples,
                    "rewards": new_rewards,
                    "dialogue_ids": new_ids,
                    "dialogues": new_dialogues,
                }, f)

            # Merge into rolling result_all
            result_all["samples"].extend(new_samples)
            result_all["rewards"].extend(new_rewards)
            result_all["dialogue_ids"].extend(new_ids)
            result_all["dialogues"].extend(new_dialogues)

        # ------ PPO training ------
        if args.rollout_only:
            print("--rollout-only: skipping TRLX training", flush=True)
            model_liar = model
            continue

        print(f"Starting TRLX training on {len(result_all['samples'])} flat samples...", flush=True)
        train_start = time.time()
        if epoch > 0:
            default_config["train"]["epochs"] = 2
            config = TRLConfig.update(default_config, {})

        # Eval prompt matches turn-1 chat-template format so eval-time generations
        # are comparable to actual rollouts.
        eval_messages = build_liar_messages(argu, prior_turns=[])
        eval_prompt = rollout_tokenizer.apply_chat_template(
            eval_messages, tokenize=False, add_generation_prompt=True
        )

                # create a dictionary of prompts and rewards
        reward_per_prompt = {}
        for i in range(len(result_all["rewards"])):
            reward = result_all["rewards"][i]
            sample = result_all["samples"][i]
            reward_per_prompt[sample] = reward
            # reward function PPO wrapper
        
        #TODO: experiment with different reward functions
        def reward_fn(samples, prompts, outputs, tokenizer, **kwargs):
            rewards = []
            for sample in samples:
                if sample in reward_per_prompt:
                    rewards.append(reward_per_prompt[sample])
                else:
                    rewards.append(0.0)
            return rewards
        # build up the prompts based on the dialogue history so far
        prompts = []
        for diologue in result_all["dialogues"]:
            # reuse this so that we can add the new turn to the list of previous turns
            previos_turs = []
            for turn in diologue["turns"]:
                liar_message = build_liar_messages(diologue["argu"], prior_turns=previos_turs)
                prompt = rollout_tokenizer.apply_chat_template(
                    liar_message, tokenize=False, add_generation_prompt=True,
                )
                previos_turs.append(turn)
                prompts.append(prompt)
        
        # promts: diologue history so far
        # output: the liar's response (on the current turn)
        # reward: the reward for the liar's response

        liar = trlx.train(
            reward_fn=reward_fn,
            prompts=prompts,
            config=config,
            eval_prompts=[eval_prompt] * 2,
        ).model

        print(f"TRLX training finished in {(time.time() - train_start)/60:.2f} minutes", flush=True)

        if ave_reward > best_reward:
            base = "/checkpoints" if os.path.isdir("/checkpoints") else "."
            save_path = f"{base}/ckpts_liar_case_{ave_reward}_axiom_{arg1}_turns{args.max_turns}"
            liar.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            best_reward = ave_reward
            print(f"Saved checkpoint to {save_path}", flush=True)

        # TRLX wraps the model; re-use the underlying base for next epoch's
        # generation (matches original behavior at the bottom of the old loop).
        model_liar = liar.base_model
        print(f"Finished Epoch {epoch + 1} in {(time.time() - epoch_start)/60:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
