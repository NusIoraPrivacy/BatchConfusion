import json
import os

current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))
data_name = "medical_o1_reasoning_SFT"
in_file_name = "compress_fake_qattr.json"

# with open(f'{root_path}/data/{data_name}/{in_file_name}') as fin:
#     data = json.load(fin)
with open(f'{root_path}/result/{data_name}/{in_file_name}') as fin:
    data = json.load(fin)
# print(data[0])
n_org_attrs = 0
n_cpr_attrs = 0
n_fakes = []
for sample in data:
    org_attrs = sample['filtered private attributes question']
    cpr_attrs = sample['filtered private attributes compression']
    n_org_attrs += len(org_attrs)
    n_cpr_attrs += len(cpr_attrs)
    fake_attrs = sample['fake attributes question']
    # fake_attrs = sample['fake attributes compression']
    if len(fake_attrs) > 0:
        n_fakes.append(len(fake_attrs))
    # for fake_list in fake_attrs:
    #     n_fakes.append(len(fake_list))


n_org_attrs /= len(data)
n_cpr_attrs /= len(data)
n_fakes = sum(n_fakes)/len(n_fakes)
print("number of origin attributes:", n_org_attrs)
print("number of compressed attributes:", n_cpr_attrs)
print("number of fake attributes:", n_fakes)