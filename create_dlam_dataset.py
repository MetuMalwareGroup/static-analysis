#from gensim.models import word2vec
from bin2op import parse, unique, counts, nextIndex
# import numpy as np
import math
import sys, os
import argparse
import pickle
# np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--destination')
parser.add_argument('-s', '--source')
parser.add_argument('-p', '--points')
args = parser.parse_args()
points = False
if args.destination:
    destination = args.destination
else:
    print('need a destination directory parameter')
    sys.exit(0)
if args.source:
    source = args.source
else:
    print('need a source directory parameter')
    sys.exit(0)
if args.points:
    points = True
    print('adding dots to end of instructions')


for file in os.listdir(source):
    try:
        syntax = "intel"
        print('file process started > {}'.format(file))
        shellcode, code, opcodes, operands, instructions = parse(os.path.join(source,file), syntax, None)

        sentences = instructions

        with open(os.path.join(destination+file+'.txt'), 'w') as sentences_file:
            #print(sentences)
            # pickle.dump(sentences, sentences_file)
            # for sentence in sentences:
            # sentences_file.writelines(sentences)
            if points:
                for l in sentences:
                    l +=  "."
                    sentences_file.write(l +"\n")
            else:    
                sentences_file.writelines("%s\n" % l for l in sentences)
            print('file processed > {}'.format(file))
    except Exception as e:
        print('file couldnt be processed > {} {} '.format(file, str(e)))

