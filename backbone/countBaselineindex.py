# %% [makrdown]
import os
baseline_index_path = "/home/gqxwolf/mydata/projectData/BackBone/indexes/baseline_2000_4_3"
index_list = os.listdir(baseline_index_path)


# %%
def countline(filename):
    lines = []
    with open(filename) as f:
        lines = f.readlines()
    return len(lines)


# %%
total_line = 0
linenumber = 0
for idx_file in index_list:
    idx_path = baseline_index_path+"/"+idx_file
    size = countline(idx_path)
    if(idx_file=="0.idx"):
        print(size, '', idx_path)
    total_line = total_line+size
    linenumber += 1

print(linenumber, '', total_line)
# %%


# %%
