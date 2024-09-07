import subprocess

def run_training():
    command = ["python", "-m", "levanter.main.train_lm", '--data.train_urls', '["/opt/ml/code/data/jd1_drums/tokenized-events-Train.txt"]', '--data.validation_urls', '["/opt/ml/code/data/jd1_drums/tokenized-events-Validation.txt"]', '--data.tokenizer', 'passthrough', '--data.plaintext', 'true', '--data.vocab_size', '55028', '--data.enforce_eos', 'false', '--model.type', 'gpt2', '--model.hidden_dim', '768', '--model.num_heads', '12', '--model.num_layers', '12', '--model.seq_len', '1024', '--model.scale_attn_by_inverse_layer_idx', 'true', '--model.embed_pdrop', '0.1', '--model.resid_pdrop', '0.1', '--model.gradient_checkpointing', 'true', '--initialize_from_hf', '"stanford-crfm/music-small-800k"', '--trainer.mp', '"p=f32,c=bfloat16"', '--trainer.model_axis_size', '1', '--trainer.per_device_parallelism', '16', '--trainer.num_train_steps', '2001', '--trainer.train_batch_size', '512', '--trainer.per_device_eval_parallelism', '1', '--trainer.checkpointer.base_path', '"/opt/ml/code/checkpoints"', '--trainer.checkpointer.save_interval', '"30m"', '--trainer.checkpointer.keep', '[{"every": 1000}]', '--trainer.axis_resources', '{"batch": "data", "vocab": "model", "mlp": "model", "heads": "model"}', '--trainer.parameter_axis_resources', '{"embed": "data"}', '--optimizer.learning_rate', '"3E-5"', '--optimizer.weight_decay', '0.1', '--optimizer.min_lr_ratio', '0.1']
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='')
    process.wait()

if __name__=="__main__":
    run_training()

