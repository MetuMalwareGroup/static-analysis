import os 
import pickle
import argparse

p_max_sentences = 6000000
cur_sentences_count = 0
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory')
parser.add_argument('-f', '--file')
parser.add_argument('-l', '--label')
parser.add_argument('-m', '--maxsentences')

args = parser.parse_args()

if args.directory:
    directory = args.directory
else:
    print('needs a directory parameter')
    sys.exit(0)

if args.file:
    p_file = args.file
else:
    print('needs a file parameter.')
    sys.exit(0)

if args.label:
    p_label = args.label
else:
    print('needs a label (0,1 for malw,bngn) parameter.')
    sys.exit(0)

if args.maxsentences:
    p_max_sentences = args.max_sentences
else:
    print('Using Default max sentence 6000000.')



with open(p_file, 'w') as all_asms:
  for file in os.listdir(directory):
    with open(os.path.join(directory,file), 'rb') as filehandle:
      sentences = pickle.load(filehandle,encoding='utf8')
      for sentence in sentences:
        cur_sentences_count += 1
        if cur_sentences_count >= p_max_sentences:
          break
        else:
          all_asms.write(sentence +"; " + p_label +'\n')