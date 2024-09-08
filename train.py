import argparse
import subprocess
import random
import string
import os
import json

def run_training():
    # SET UP ID
    ID = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    if "SM_TRAINING_ENV" in os.environ:
        ID = json.loads(os.environ["SM_TRAINING_ENV"])["job_name"]
    
    # SET UP TRAINING URLS
    if "SM_CHANNEL_TRAIN" not in os.environ:
        print("No SM_CHANNEL_TRAIN found in environment.")
        return 1
    train_urls = [os.path.join(os.environ["SM_CHANNEL_TRAIN"], x) for x in os.listdir(os.environ["SM_CHANNEL_TRAIN"])]

    # SET UP VALIDATION URLS
    if "SM_CHANNEL_VALIDATION" not in os.environ:
        print("No SM_CHANNEL_VALIDATION found in environment.")
        return 1
    validation_urls = [os.path.join(os.environ["SM_CHANNEL_VALIDATION"], x) for x in os.listdir(os.environ["SM_CHANNEL_VALIDATION"])]

    # SET UP CHECKPOINT DIR
    if "SM_MODEL_DIR" not in os.environ:
        print("No SM_MODEL_DIR found in environment.")
        return 1 

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)

    args = parser.parse_args()

    command = ["python", "-m", "levanter.main.train_lm", '--data.train_urls', str(train_urls), '--data.validation_urls', str(validation_urls), '--data.tokenizer', 'passthrough', '--data.plaintext', 'true', '--data.vocab_size', '55028', '--data.enforce_eos', 'false', '--model.type', 'gpt2', '--model.hidden_dim', '768', '--model.num_heads', '12', '--model.num_layers', '12', '--model.seq_len', '1024', '--model.scale_attn_by_inverse_layer_idx', 'true', '--model.embed_pdrop', '0.1', '--model.resid_pdrop', '0.1', '--model.gradient_checkpointing', 'true', '--initialize_from_hf', '"stanford-crfm/music-small-800k"', '--trainer.mp', '"p=f32,c=bfloat16"', '--trainer.model_axis_size', '1', '--trainer.per_device_parallelism', '16', '--trainer.id', ID, '--trainer.wandb.project', 'jordanai-aws-levanter', '--trainer.num_train_steps', '2001', '--trainer.train_batch_size', args.batch_size, '--trainer.per_device_eval_parallelism', '1', '--trainer.checkpointer.base_path', os.environ["SM_MODEL_DIR"], '--trainer.checkpointer.save_interval', '"30m"', '--trainer.checkpointer.keep', '[{"every": 1000}]', '--trainer.axis_resources', '{"batch": "data", "vocab": "model", "mlp": "model", "heads": "model"}', '--trainer.parameter_axis_resources', '{"embed": "data"}', '--optimizer.learning_rate', '"3E-5"', '--optimizer.weight_decay', '0.1', '--optimizer.min_lr_ratio', '0.1']

    print(command)

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='')
    process.wait()

    return process.returncode

if __name__=="__main__":
    exit(run_training())

