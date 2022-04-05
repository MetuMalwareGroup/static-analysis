# this python file takes files from dlam dataset and
# creates slam dataset
import os 

ORG_DATA_DIR="./dataset_dlam/"
ORG_TRAIN_MAL_DIR = os.path.join(ORG_DATA_DIR,'train/malw')
ORG_TRAIN_BNGN_DIR = os.path.join(ORG_DATA_DIR,'train/bngn')
ORG_TEST_MAL_DIR =os.path.join(ORG_DATA_DIR,'test/malw')
ORG_TEST_BNGN_DIR = os.path.join(ORG_DATA_DIR,'test/malw')

PRO_DATA_DIR="./dataset_slam/"
PRO_TRAIN_MAL_DIR = os.path.join(PRO_DATA_DIR,'train/malw')
PRO_TRAIN_BNGN_DIR = os.path.join(PRO_DATA_DIR,'train/bngn')
PRO_TEST_MAL_DIR =os.path.join(PRO_DATA_DIR,'test/malw')
PRO_TEST_BNGN_DIR = os.path.join(PRO_DATA_DIR,'test/bngn')

orgarray = [ORG_TRAIN_MAL_DIR,ORG_TRAIN_BNGN_DIR,ORG_TEST_MAL_DIR,ORG_TEST_BNGN_DIR]
proarray = [PRO_TRAIN_MAL_DIR,PRO_TRAIN_BNGN_DIR,PRO_TEST_MAL_DIR,PRO_TEST_BNGN_DIR]

for oa,pa in zip(orgarray,proarray):
    for f in os.listdir(oa) :
        count = 0
        with open(os.path.join(oa,f), 'rb') as filehandle:
            while True:
                # Get next line from file
                line = filehandle.readline()
                # if line is empty
                # end of file is reached
                if not line:
                    break
                if b"(bad)" in line or b"align" in line \
                    or b"nop" in line or b"aaa" in line \
                        or b"int3" in line:
                    continue
                else:
                    count += 1
                with open(os.path.join(pa,str(count)+f), 'wb') as single_line_file:
                    single_line_file.write(line)
        print(oa, f)

