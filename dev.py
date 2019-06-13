import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


import random

import tools



if __name__ == '__main__':

    tools.set_seed(0)

    sketch_root_folder = os.path.join(HERE_PATH, 'samples')
    sketch_folders = tools.list_folders(sketch_root_folder)


    for i in range(1):

        type1_folder = random.choice(sketch_folders)

        type2_folder = random.choice(sketch_folders)
        while type2_folder == type1_folder:
            type2_folder = random.choice(sketch_folders)


        type1_files = tools.list_files(type1_folder, ['*.json'])
        type2_files = tools.list_files(type2_folder, ['*.json'])


        print(type1_files[0])
        data = tools.read_json(type1_files[0])
        print(data)
