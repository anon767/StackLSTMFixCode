from sctokenizer import CTokenizer
import glob
import re 
import pickle


all_files = glob.glob("dataset/*.c")
filename_pattern = r"(CVE_\d{4}_\d{4})"
file_set = set([re.findall(filename_pattern, x)[0] for x in all_files if re.findall(filename_pattern, x) ])

tokenizer = CTokenizer() # this object can be used for multiple source files
vocabulary = set()
dataset = []
for file in file_set:
    patched_file = glob.glob(f"dataset/{file}_PATCHED*.c")[0]
    vuln_file = glob.glob(f"dataset/{file}_VULN*.c")[0]
    temp_dict = {"name":file}
    for curr_file, tar_or_inp in zip([vuln_file, patched_file], ["x", "y"]):
        with open(curr_file) as f:
            source = f.read()
            all_tokens = tokenizer.tokenize(source)
            tokens = [x.token_value for x in all_tokens] # save only a token (token_type and line are dropped). https://pypi.org/project/sctokenizer/ 
            vocabulary.update(tokens)
            temp_dict[tar_or_inp] = tokens
    
    dataset.append(temp_dict)

with open('dataset_preprocessed.pkl', 'wb') as handle:
    pickle.dump(dataset, handle) 

with open('vocab.pkl', 'wb') as handle:
    pickle.dump(vocabulary, handle) 
