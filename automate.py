import time
import threading
import os
from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

def getFiles(path):
    list_files = []
    for file in os.listdir(path):
        list_files.append(os.path.join(path, file))

    return list_files

def submitFile(path_to_file):
    
    cmd = 'cuckoo submit --platform windows --priority 5 {file}'.format(file = path_to_file)
    os.system(cmd)
    return True

# class OnMyWatch:
#     # Set the directory on watch
#     watchDirectory = "/home/cuckoo/.cuckoo/storage/analyses"

#     def __init__(self):
#         self.observer = Observer()

#     def run(self):
#         event_handler = Handler()
#         self.observer.schedule(event_handler, self.watchDirectory, recursive = True)
#         self.observer.start()
#         try:
#             while True:
#                 time.sleep(5)
#         except:
#             self.observer.stop()
#             print("Observer Stopped")

#         self.observer.join()

# class Handler(FileSystemEventHandler):

#     @staticmethod
#     def on_any_event(event):
#         if event.is_directory:
#             return None

#         elif event.event_type == 'created':
#             # Event is created, you can process it now
#             if event.src_path.endswith('report.json') == True:
#                 print (event.src_path)
#                 num_files = num_files + 1
#                 if submitFile(list_files[num_files]):
#                     print ("Succesfully submit {current}/{total}".format(current = num_files, total = len(list_files)))
#                     time.sleep(15)
#                 else:
#                     raise ("Unable to submit {file} to Cuckoo.".format(file = list_files[num_files]))

#         elif event.event_type == 'modified':
#             # Event is modified, you can process it now
#             pass


original_path = '/home/containernet/KLTN/Pesidious/independent/malware/mutate'
# decision = '/home/containernet/KLTN/Pesidious/Mutated_DecisionTree'
random = '/home/containernet/KLTN/Pesidious/Mutated_RandomForest'
gradient = '/home/containernet/KLTN/Pesidious/Mutated_Gradient'
stacking = '/home/containernet/KLTN/Pesidious/Mutated_StackingRandomForest'
kneighbors = '/home/containernet/KLTN/Pesidious/Mutated_10_actions/Mutated_KNeighbors'
gradient_old = '/home/containernet/KLTN/Pesidious/Mutated_10_actions/Mutated_Gradient_Old'

def func(string):
    return 'mutated_' + string

with open('/home/containernet/KLTN/final_file.txt', 'r') as f:
    list_file = f.read().split('\n')[:-1]

mutated = list(map(func, list_file))

# print ("Submit original samples!")
# for hash in tqdm(list_file, desc="File: "):
#     print ('\n')
#     submitFile(os.path.join(original_path, hash))
#     time.sleep(90)

print ("Submit mutated samples!")
for folder in tqdm(([gradient_old]), desc="Folder: "):
    for i in tqdm(range(100), desc="File: "):
        print('\n')
        submitFile(os.path.join(folder, mutated[i]))
        time.sleep(80)

print ("ALL DONE!")
