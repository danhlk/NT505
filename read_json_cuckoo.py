from __future__ import print_function
import json
import os

from pathlib import Path
from tqdm import tqdm

path = "/home/cuckoo/.cuckoo/storage/analyses"

result_file = Path('compare_result.txt')
headers = ",".join(["ID", "Name", "Original Signature", "Original Score", "Decision", "Decision Score", "Random Forest", "Random Forest Score", "Gradient", "Gradient Score", "KNeigbors", "KNeighbors Score", "Stacking", "Stacking Score",])
exists = result_file.exists()


with open('max.txt', 'r') as f:
    list_id = f.read().split('\n')[:-1]

list_decision = list(range(301, 401))
list_random = list(range(401, 501))
list_gradient = list(range(501, 601))
list_kneighbors = list(range(704, 804))
list_stacking = list(range(601, 701))

def f(a):
    string = ''
    for i in range(len(a["signatures"])):
        string += str(a["signatures"][i]["description"]) + '\n'
    
    return '"' + string + '"'
    # return str(len(a["signatures"]))

with open(str(result_file), 'a+') as file:
    if not exists:
         file.write(headers)
    
    for i in tqdm(range(0, 100), desc="Progress:"):
        result = list()
        data = json.load(open(os.path.join(path, '{number}/reports/report.json'.format(number=list_id[i])), 'r'))
        result.append('\n' + list_id[i])
        result.append(str(data["target"]["file"]["name"]))
        result.append(f(data))
        result.append(str(data["info"]["score"]))
        del data
        
        data = json.load(open(os.path.join(path, '{number}/reports/report.json'.format(number=list_decision[i])), 'r'))            
        result.append(f(data))
        result.append(str(data["info"]["score"]))
        del data
        
        data = json.load(open(os.path.join(path, '{number}/reports/report.json'.format(number=list_random[i])), 'r'))            
        result.append(f(data))
        result.append(str(data["info"]["score"]))
        del data
        
        data = json.load(open(os.path.join(path, '{number}/reports/report.json'.format(number=list_gradient[i])), 'r'))            
        result.append(f(data))
        result.append(str(data["info"]["score"]))
        del data

        data = json.load(open(os.path.join(path, '{number}/reports/report.json'.format(number=list_kneighbors[i])), 'r'))            
        result.append(f(data))
        result.append(str(data["info"]["score"]))
        del data

        data = json.load(open(os.path.join(path, '{number}/reports/report.json'.format(number=list_stacking[i])), 'r'))            
        result.append(f(data))
        result.append(str(data["info"]["score"]))
        del data

        file.write(",".join(result))
        del result


print ("ALL DONE!")
