
# **The custom pre-trained model based on GPT-2**

1. Use ***Documents as a sequence*** dataset to obtain  [own_gpts2_pre_trained_outputs folder](../tree/main/own_gpt2-pre_trained_outputs) using [weight_initialization.py](./main/weight_initialization.py)
2. Create vocab.json and merges.txt files using

```
paths = [str(x) for x in Path(".").glob( 'Documents_as_a_sequence.txt')]
for p in paths:
tokenizer = ByteLevelBPETokenizer()
 
tokenizer.train(files=paths, vocab_size=50_254, min_frequency=2, special_tokens=[
   "<s>",
   "<pad>",
   "</s>",
   "<unk>",
   "<mask>",
])
```
3. Save vocab.json and merges.txt files

```
tokenizer.save_model("/content/drive/My Drive/GPT2/own_gpt2-tokenize")
```
4. Use these outputs to create [own_gpts2_pre_trained_outputs folder](./own_gpt2-pre_trained_outputs) with

```
training_args = TrainingArguments(
   output_dir="own_gpts2_pre_trained_outputs folder",
   overwrite_output_dir=True,
   num_train_epochs=2,
   per_device_train_batch_size=32,
   per_device_eval_batch_size=64,
   save_steps=100_000,
   save_total_limit=2,
   prediction_loss_only=True,
)
 
trainer = Trainer(
   model=model,
   args=training_args,
   data_collator=data_collator,
   train_dataset=dataset,
)
```
5. Save outputs with 
```
trainer.save_model(own_gpts2_pre_trained_outputs folder)

```
# **DSLM-GPT2**

1. Use ***Sentences as a sequence*** dataset to feed the model.
2. Use own_gpts2_pre_trained_outputs folder files to create model using [binary_classification-own_GPT-2.py](./binary_classification_own_GPT-2.py).


# **GLM-GPT2**

1. Use ***Sentences as a sequence*** dataset to feed the model.
2. Download the pre_trained_outputs folder of GPT-2 ‘s with
```
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt-2', num_labels=2)

model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt-2', config=model_config)

tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt-2')
```
3. Create model using [binary_classification-GPT-2.py](./binary_classification-GPT-2.py) with model_config, model, and tokenizer.

# **BBUM **

1. Use ***Sentences as a sequence*** dataset to feed the model.
2. Download pre_trained_outputs folder of BERT‘s using
```
model_config = BERTConfig.from_pretrained(pretrained_model_name_or_path='bert-base-uncase', num_labels=2)

model = BERTForSequenceClassification.from_pretrained(pretrained_model_name_or_path='bert-base-uncase', config=model_config)

tokenizer = BERTTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncase')
```
3. Create model using [binary_classification-BERT.py](./binary_classification-BERT.py) using model_config, model, and tokenizer.




