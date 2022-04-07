# Creating dataset for DLAM
1. Our study extracts assembly code using an open-source disassembler to create an opcode sequence as output.  
  1.1.  Extract asssembly instructions with [extract_assembly.py](https://github.com/MetuMalwareGroup/static-analysis/blob/main/extract_assembly.py) which internally uses [bin2op.py](https://github.com/MetuMalwareGroup/static-analysis/blob/main/bin2op.py)  
  1.2.  Save asssembly intsructions in seperate files.  
  1.3.  These actions may be done using `python extract_assembly.py --source <Dir_With_PE32_Files> --destination <Dir_to_Save_Output>`  
2. As shown in ![figure](https://github.com/MetuMalwareGroup/static-analysis/blob/main/pipeline.jpeg) we merge 
4. Third item
5. Our study extracts assembly code using an open-source disassembler to create an opcode sequence as output. We used the output as our raw data to create a language model assisted with word embedding, just like processing natural language. Using this language model, we aim to adopt polarity detection methods to identify the intention of an executable file using the labels as malicious and benign. Hence, we plan to detect whether it is malicious or benign with our proposed language model.

static-analysis
Firstly,you should use bin2op.py for converting binary files to opcode file.Then, in first project,...


And for the second project, you can use own_gpts2_pre_trained_outputs for weight initialization to trained binary_classification-own_GPT-2.py.
For  binary_classification-GPT-2.py.
