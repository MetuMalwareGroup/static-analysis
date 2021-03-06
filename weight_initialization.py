# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive/')


from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path(".").glob( './drive/My Drive/GPT2/merged.txt')]
for p in paths:
  print(p)
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training special token 
tokenizer.train(files=paths, vocab_size=50_254, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

#special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}

tokenizer.save_model("/content/drive/My Drive/GPT2/own_gpt2-tokenize")

DATA_DRIVE="/content/drive/My Drive/GPT2/own_gpt2-tokenize"

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing


tokenizer = ByteLevelBPETokenizer(
    DATA_DRIVE+"/vocab.json",
    DATA_DRIVE+"/merges.txt",
)



from transformers import GPT2Config

config = GPT2Config(
    vocab_size=50_257,
    activation_function="gelu_new",
    n_positions=512,
    n_ctx=512,
)
#n_positions=1024, n_ctx=1024, 
#  "activation_function": "gelu_new",
#  "attn_pdrop": 0.1,
#  "bos_token_id": 50256,
#  "embd_pdrop": 0.1,
#  "eos_token_id": 50256,
#  "gradient_checkpointing": false,
#  "initializer_range": 0.02,
#  "layer_norm_epsilon": 1e-05,
#  "model_type": "gpt2",
#  "n_ctx": 1024,
#  "n_embd": 768,
#  "n_head": 12,
#  "n_inner": null,
#  "n_layer": 12,
#  "n_positions": 1024,
#  "resid_pdrop": 0.1,
#  "scale_attn_weights": true,
#  "summary_activation": null,
#  "summary_first_dropout": 0.1,
#  "summary_proj_to_labels": true,
#  "summary_type": "cls_index",
#  "summary_use_proj": true,
#  "transformers_version": "4.6.0.dev0",
#  "use_cache": true,
#  "vocab_size": 50257

from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("/content/drive/My Drive/GPT2/own_gpt2-tokenize", max_len=256)

special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

model.resize_token_embeddings(len(tokenizer))

from transformers import GPT2LMHeadModel #RobertaLMHeadModel??

model = GPT2LMHeadModel(config=config)

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/content/drive/My Drive//merged.txt",
    block_size=256,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/content/drive/My Drive/GPT2/own_gpt2-pre_trained_outputs",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("/content/drive/My Drive/GPT2/own_gpt2-pre_trained_outputs")
