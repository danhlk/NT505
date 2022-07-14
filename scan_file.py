from vt import Client
import sys
from pathlib import Path
from time import sleep
from tqdm import tqdm
import os

with open('api.txt', 'r') as api:
    api_list = api.read().split('\n')[:-1]

path = "/home/containernet/KLTN/Pesidious/independent/malware/mutate"
list_file = list()
for file in os.listdir(path):
    list_file.append(file)

result_file = Path(str(sys.argv[1]) + '_' + str(sys.argv[2]))
exists = result_file.exists()
result = list()

with open(str(result_file), 'a+') as f_out:
    header = ','.join(["Hash", "Detected",])
    if not exists:
        f_out.write(header)
    client = Client(apikey=api_list[int(sys.argv[2])])
    for i in tqdm(range(int(sys.argv[3]), int(sys.argv[4])), desc="Progress: "):
        with open(os.path.join(path, list_file[i]), 'rb') as f:
            analysis = client.scan_file(f, wait_for_completion=True)
        f_out.write(','.join(['\n' + list_file[i], f'{analysis.stats.get("malicious")}/{sum(analysis.stats.values())}']))
        sleep(17)
            

print (str(sys.argv[2]) + '_DONE!')
