#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Extract the opcode from the objdump of a binary
this code is highly influenced from : https://github.com/dhn/bin2op/blob/master/bin2op.py
'''

import re
import os
import sys
import argparse
import string
from subprocess import check_output
from collections import Counter
import re


shellcode=None
code=None
opcodes=None
operands=None

def nextIndex(needle, haystack, start: int=0):
    while True:
        try:
            start = haystack.index(needle, start)
            yield start
            start += 1
        except ValueError:
            raise StopIteration

def counts(list,verbose=False):
    #return dict((elem, list.count(elem)) for elem in list)   #this takes too much time
    if verbose:
        print(Counter(list))
    else:
        return Counter(list)
# function to get unique values 
def unique(list1,verbose=False): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    # print list 
    if verbose:
        for x in unique_list: 
            print(x, end=",") 
    else:
        return unique_list

# thanks zerosum0x0
# http://llvm.1065342.n5.nabble.com/llvm-dev-clang-triple-and-clang-target-td92435.html
# http://www.skyfree.org/linux/references/coff.pdf

def parse(obj, syntax, formats):
    try:
        objdump = ['objdump', '-d', '-x86-asm-syntax', syntax, "--triple","i386",  "--arch","x86", obj]
        lines = check_output(objdump)
        lines = lines.split(b'Disassembly of section')[1]
        lines = lines.split(b'\n')[3:]

        shellcode = ""
        code = []
        opcodes = []
        operands = []
        instructions = []
        for line in lines:
            line = line.strip()
            tabs = line.split(b'\t')
            #print("tabs {}".format(tabs))
            if (len(tabs) < 2):
                continue
            bytes = tabs[0].strip()

            instruction = ""
            if (len(tabs) == 3):
                instruction = tabs[1].strip().decode("utf-8")
                instruction += " "+tabs[2].strip().decode("utf-8")
                # instruction_clean = instruction.replace(',',' ')
                instruction_clean = re.sub(' +', ' ', instruction)
                #print("instruction {}".format(instruction))
                # instruction_clean = re.sub(r'([D|Q]{0,}WORD |BYTE )', 'A', instruction_clean)
                # instruction_clean = re.sub(r'(0x[0-9a-fA-F]+)(?:)?', 'addr', instruction_clean)
                instructions.append(instruction_clean)
                instruction_split = instruction_clean.split(' ')
                opcode = instruction_split[0] if len(instruction_split) > 0 else 'none' 
                opcodes.append(opcode)        
                operand = instruction_split[1:] if len(instruction_split) > 1 else 'none' 
                operands.extend(operand)

            bytes = bytes.split(b' ')
            shellcodeline = ""
            for byte in bytes:
                shellcodeline +=  byte.decode("utf-8") + " "

            shellcode += shellcodeline
            if formats is not None:
                c = '\t%-*s# %s' % (32, '"'+shellcodeline+'"', instruction)
            else:
                c = '%-*s/* %s */' % (32, '"'+shellcodeline+'"', instruction)
            code.append(c)
    except Exception as e:
        print(str(e))
        return None

    return shellcode, code, opcodes, operands, instructions




if __name__ == "__main__":
    if len(sys.argv) <= 1:
        usage()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--intel', action='store_true')
        parser.add_argument('-a', '--att', action='store_true')
        parser.add_argument('-o', '--opcode', dest='opcode', action='store_true')
        parser.add_argument('-u', '--uniqopcode', dest='uniqopcode', action='store_true')
        parser.add_argument('-c', '--countopcode', dest='countopcode', action='store_true')
        parser.add_argument('-p', '--operand' , dest='operand', action='store_true')
        parser.add_argument('-w', '--ops' , dest='ops', action='store_true')
        parser.add_argument('-s', '--sentence' ,dest='sentence', action='store_true')
        parser.add_argument('-f', '--file' )
        parser.add_argument('-v', dest='verbose', action='store_true')
        args = parser.parse_args()
        if args.intel:
            syntax = 'intel'
        if args.att :
            syntax = 'att'  
        if args.sentence:
            sentence = True  
        if args.file:
            if os.path.exists(args.file):
                shellcode, code, opcodes, operands, instructions = parse(args.file, syntax, None)
                if args.opcode:
                    print("-"*20+"opcodes"+"-"*20 +"\n")
                    for opcode in opcodes:
                        print(opcode, end=',')
                if args.uniqopcode:
                    print("-"*20+"unique_opcodes"+"-"*20 +"\n")
                    unique_opcodes=unique(opcodes)
                    for opcode in unique_opcodes:
                        print("'"+opcode+"'", end=',')
                    print("\n"+"-"*20+"unique_opcodes_length"+"-"*20 +"\n")
                    print(len(unique_opcodes))
                if args.countopcode:
                    print("-"*20+"count_opcodes"+"-"*20 +"\n")
                    counts(opcodes,True)
                if args.operand:
                    print("-"*20+"unique_operands"+"-"*20 +"\n")
                    unique_operands=unique(operands)
                    for operand in unique_operands:
                        print("'"+operand+"'", end=',')
                    print("\n"+"-"*20+"unique_operands_length"+"-"*20 +"\n")
                    print(len(unique_operands))
                if args.sentence:
                    print("-"*20+"instructions"+"-"*20 +"\n")
                    print("[")
                    for instruction in instructions:
                        print("'"+instruction+"'", end=',')
                    print("]")
                if args.ops:
                    print("-"*20+"opcodes and operands"+"-"*20 +"\n")
                    print("[")
                    unique_operands=unique(operands)
                    unique_opcodes=unique(opcodes)
                    ops = unique(unique_operands + unique_opcodes)
                    for op in ops:
                        print("'"+op+"'", end=',')
                    print("]")
                    print(len(ops))
            else:
                print("[!] file does not exist")
                sys.exit(1)


