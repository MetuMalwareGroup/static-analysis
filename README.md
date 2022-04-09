# The Document Level Analysis Model (The DLAM)
1. Our study extracts assembly code using an open-source disassembler to create an opcode sequence as output.  
  1.1.  Extract asssembly instructions with [create_dlam_dataset.py](./create_dlam_dataset.py) which internally uses [bin2op.py](./bin2op.py)  
  1.2.  Save asssembly intsructions in seperate files.  
  1.3.  These actions may be done using `python create_dlam_dataset.py --source <Dir_With_PE32_Files> --destination <Dir_to_Save_Output>`  
2. We name the directory that assembly outputs reside as  ***assembly pool***.
3. As shown in ![figure](./pipeline.jpeg) we processed assembly outputs with `custom_standardization` function in [custom_standardization.py](./custom_standardization.py). 
4. To create assembly pool which we will use in the DLAM, first we converted opcode sequences to vectorized form with the layer 
    ```
    vectorize_layer = TextVectorization(
      standardize=custom_standardization,
      max_tokens=max_features,
      output_mode='int',
      pad_to_max_tokens=True,
      ngrams=ngrams,
      output_sequence_length=sequence_length
      )
    ```   
      This layer, after adaptation `vectorize_layer.adapt(train_text)` gives us 
    ```
      Vectorized opcode sequences (<tf.Tensor: shape=(1, 1024),   dtype=int64, numpy=array([[111,  17,  40, ...,  94,  79,    82]])>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
    ```
     
4. The training, validation and testing dataset seperation and preparation may be seen in [DLAM_TEXT_CLASSIFICATION.html](./00_of_text_classification_dlam.html). 
5.  The DLAM which we constructed as  
    ```
    model = tf.keras.Sequential([
    layers.Embedding(max_features+1, embedding_dim),
    layers.Bidirectional(layers.LSTM(128,dropout = 0.5, recurrent_dropout = 0.5, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(128,dropout = 0.5, recurrent_dropout = 0.5, return_sequences=True)),
    layers.Dropout(0.5),
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.5),
    layers.Dense(128),
    layers.Dropout(0.5),
    layers.Dense(1)])
    ```

# The Sentence Level Analysis Model (The SLAM)
1.  We split the assembly instructions in assembly pool with [prepare_slam_data.py](./prepare_slam_data.py) into single line containing files.
2. The files form our assembly pool which we use sentences as sequences for the SLAM.  
3.  We constructed The SLAM  
    3.1 With n-grams as:
    ```
    model = tf.keras.Sequential([
    layers.Embedding(max_features+1, embedding_dim),
    layers.Bidirectional(layers.LSTM(128,dropout = 0.5, recurrent_dropout = 0.5, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(128,dropout = 0.5, recurrent_dropout = 0.5, return_sequences=True)),
    layers.Dropout(0.5),
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.5),
    layers.Dense(128),
    layers.Dropout(0.5),
    layers.Dense(1)])
    ```
    3.2 With Word2Vec as: 
    ```
    model = Sequential()  
    model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_size,
                    weights=[w2v_weights],
                    input_length=MAX_SEQUENCE_LENGTH,
                    mask_zero=True,
                    trainable=False))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128,)))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    ```
    3.3 With DistilBERT as:
    ```
    def build_model(transformer, max_length=params['MAX_LENGTH']):
    
    
    # Define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=params['RANDOM_STATE']) 
    
    # Define input layers
    input_ids_layer = tf.keras.layers.Input(shape=(max_length,), 
                                            name='input_ids', 
                                            dtype='int32')
    input_attention_layer = tf.keras.layers.Input(shape=(max_length,), 
                                                  name='input_attention', 
                                                  dtype='int32')
    
    # DistilBERT outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).

    last_hidden_state, ikinci = transformer([input_ids_layer, input_attention_layer])    
    
    B1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(last_hidden_state)
    
    B2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(B1)

    D1 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                 seed=params['RANDOM_STATE']
                                )(B2)
    M1 = tf.keras.layers.GlobalMaxPooling1D()(D1)
    D2 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                 seed=params['RANDOM_STATE']
                                )(M1)
    X = tf.keras.layers.Dense(128,
                              activation='relu',
                              kernel_initializer=weight_initializer,
                              bias_initializer='zeros'
                              )(D2)
    
    D3 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                 seed=params['RANDOM_STATE']
                                )(X)
    
    # # Define a single node that makes up the output layer (for binary classification)
    output = tf.keras.layers.Dense(1, 
                                   activation='sigmoid',
                                   kernel_initializer=weight_initializer,  
                                   bias_initializer='zeros'
                                   )(D3)
    
    # Define the model
    model = tf.keras.Model([input_ids_layer, input_attention_layer], output)
    
    # Compile the model
    model.compile(tf.keras.optimizers.Adam(lr=params['LEARNING_RATE']), 
                  loss=focal_loss(),
                  metrics=['accuracy'])
    
    return model
    ```

    3.4 The Custom Pre-Trained Model Based on GPT-2  
    3.4.1. Use ***Documents as a sequence*** dataset to obtain  [our_gpt2_pre_trained_outputs](./our_gpt2-pre_trained_outputs/) using [weight_initialization.py](./weight_initialization.py)  
    3.4.2. Create vocab.json and merges.txt files using  

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
    3.4.3. Save vocab.json and merges.txt files  
    ```
    tokenizer.save_model("/content/drive/My Drive/GPT2/own_gpt2-tokenize")
    ```
    3.4.4. Use these outputs to create [our_gpt2_pre_trained_outputs](./our_gpt2-pre_trained_outputs) with  

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
    3.4.5. Save outputs with  
    ```
    trainer.save_model(our_gpts2_pre_trained_outputs)

    ```

    3.5. GPT2 Domain Specific Language Model (GPT2-DSLM)

    3.5.1. Use ***Sentences as a sequence*** dataset to feed the model.  
    3.5.2. Use own_gpts2_pre_trained_outputs folder files to create model using [binary_classification-own_GPT-2.py](./binary_classification_own_GPT-2.py).


    3.6. GPT2 General Language Model (GPT2-GLM)  
    3.6.1. Use ***Sentences as a sequence*** dataset to feed the model.  
    3.6.2. Download the pre_trained_outputs folder of GPT-2 ‘s with
    ```
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt-2',    num_labels=2)

    model = GPT2ForSequenceClassification.from_pretrained   (pretrained_model_name_or_path='gpt-2', config=model_config)

    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt-2')
    ```
    3.6.3. Create model using [binary_classification-GPT-2.py](./binary_classification-GPT-2.py) with model_config, model, and tokenizer.

    3.7. BERT General Language Model (BERT-GLM)  
    3.7.1. Use ***Sentences as a sequence*** dataset to feed the model.  
    3.7.2. Download pre_trained_outputs folder of BERT‘s using
    ```
    model_config = BERTConfig.from_pretrained(pretrained_model_name_or_path='bert-base-uncase', num_labels=2)

    model = BERTForSequenceClassification.from_pretrained(pretrained_model_name_or_path='bert-base-uncase', config=model_config)

    tokenizer = BERTTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncase')
    ```
    3.7.3. Create model using [binary_classification-BERT.py](./binary_classification-BERT.py) using model_config, model, and tokenizer.




