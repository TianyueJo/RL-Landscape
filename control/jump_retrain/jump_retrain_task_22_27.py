#!/usr/bin/env python3
"""
Run Jump & Retrain for controlled_task_22 and controlled_task_27
- step sizes: 10, 20, 30
- 5 different random seeds per step size
- Train for 0.5M steps
- Total runs: 2 * 3 * 5 = 30
"""

import sys
from pathlib import Path

# Import functions from jump_and_retrain_rl
from jump_and_retrain_rl import (
    load_base_task_metadata,
    run_single_jump_and_retrain,
)

def main():
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    config_path = base_dir / "config" / "ppo_lstm_chatgpt.yml"
    
    task_ids = [22, 27]
    step_sizes = [20, 40, 60]
    num_seeds = 5
    extra_steps = 500_000
    
    # Determine env name (HalfCheetah-v4, since tasks 22/27 are in 16-31)
    env_name = "HalfCheetah-v4"
    
    # Output directories
    output_root = base_dir / "results" / "jump_retrain_22_27"
    models_root = base_dir / "models" / "jump_retrain_22_27"
    
    total_runs = len(task_ids) * len(step_sizes) * num_seeds
    current_run = 0
    
    print("Starting Jump & Retrain")
    print(f"Tasks: {task_ids}")
    print(f"Step sizes: {step_sizes}")
    print(f"Random seeds per step size: {num_seeds}")
    print(f"Training steps: {extra_steps:,}")
    print(f"Total runs: {total_runs}")
    print(f"Output dir: {output_root}")
    print(f"Models dir: {models_root}")
    print("=" * 70)
    
    # Global jr_index to ensure each run has a unique index
    global_jr_index = 0
    
    for task_id in task_ids:
        base_task_dir = models_dir / f"controlled_task_{task_id}"
        if not base_task_dir.exists():
            print(f"⚠ Skip task {task_id}: directory does not exist: {base_task_dir}")
            continue
        
        # Load base task metadata
        base_meta = load_base_task_metadata(base_task_dir)
        if base_meta["env_name"] != env_name:
            print(f"⚠ Skip task {task_id}: env mismatch ({base_meta['env_name']} != {env_name})")
            continue
        
        # Select best_model (fallback to final_model)
        base_model_path = base_task_dir / "best_model.pt"
        if not base_model_path.exists():
            base_model_path = base_task_dir / "final_model.pt"
            if not base_model_path.exists():
                print(f"⚠ Skip task {task_id}: model file not found")
                continue
        
        print(f"\n[Base Task {task_id}]")
        print(f"  best_reward: {base_meta['best_reward']:.2f}")
        print(f"  model: {base_model_path.name}")
        
        for step_size in step_sizes:
            for seed_idx in range(num_seeds):
                current_run += 1
                jr_index = global_jr_index
                global_jr_index += 1
                
                # Use a different rng_seed for each combination
                rng_seed = 10000 + jr_index
                
                print(f"\n[{current_run}/{total_runs}] Run: task={task_id}, step_size={step_size}, seed_idx={seed_idx}")
                print(f"  jr_index: {jr_index}, RNG seed: {rng_seed}")
                
                try:
                    run_single_jump_and_retrain(
                        base_task_id=str(task_id),
                        base_task_dir=base_task_dir,
                        base_meta=base_meta,
                        base_model_path=base_model_path,
                        config_path=config_path,
                        env_name=env_name,
                        device="cuda",
                        step_size=step_size,
                        jr_index=jr_index,
                        extra_steps=extra_steps,
                        rng_seed=rng_seed,
                        output_root=output_root,
                        models_root=models_root,
                    )
                    print(f"  ✓ Done: task={task_id}, step_size={step_size}, seed_idx={seed_idx}")
                except Exception as e:
                    print(f"  ✗ Failed: task={task_id}, step_size={step_size}, seed_idx={seed_idx}")
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with the next run
                    continue
    
    print("\n" + "=" * 70)
    print("All runs completed!")
    print(f"Results saved to: {output_root}")
    print(f"Models saved to: {models_root}")


if __name__ == "__main__":
    main()

