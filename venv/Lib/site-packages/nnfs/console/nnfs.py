import os
import sys
import shutil
import nnfs_video_tutorial_code


def print_usage():
    print('''
Neural Networks from Scratch in Python Tool.

Basic usage:
nnfs command [parameter1 [parameter2]]

Detailed usage:
nnfs info | code video_part [destination]

Commands:
  info    Prints information about the book
  code    Creates a file containing the code of given video part
          in given location. Location is optional, example:
          nnfs code 2 nnfs/p02.py
          will create a file p02.py in a nnfs folder containing
          the code of part 2 of video tutorial
    ''')


def info():
    print('''
  Neural Networks from Scratch in Python
  by Harrison Kinsley & Daniel Kukieła

  https://nnfs.io/

  ---

  This package contains the code and supplementary material
  as well as lesson code related to the book and video series.

    ''')


def code(video_part, path='/.'):

    final_path = path.lstrip("/").lstrip("\\")
    final_path = os.path.realpath(os.getcwd() + '/' + final_path)
    if path.rstrip('.').endswith('/') or path.rstrip('.').endswith('\\'):
        final_path += '/'

    try:
        int(video_part)
    except:
        print('Video part needs to be an integer')
        return

    code_folder = os.path.dirname(nnfs_video_tutorial_code.__file__)

    parts = [int(file.split('.')[0]) for file in os.listdir(code_folder) if os.path.isfile(code_folder + '/' + file) and not file.startswith('_')]
    if int(video_part) > max(parts):
        print('Video part does not exist')
        return

    try:
        os.makedirs(final_path if path.rstrip('.').endswith('/') or path.rstrip('.').endswith('\\') else os.path.dirname(final_path), exist_ok=True)
    except:
        print('Could not create one on more directories in a treefile ' + final_path + ' - permission issue?')

    if path.rstrip('.').endswith('/') or path.rstrip('.').endswith('\\'):
        final_path += 'p{:03d}.py'.format(int(video_part))

    try:
        shutil.copyfile(code_folder + '/{:03d}.py'.format(int(video_part)), final_path)
    except:
        print('Could not create file ' + final_path + 'permission issue?')

    print('Tutorial code for part ' + video_part + ' saved as ' + final_path)


def main():

    if len(sys.argv) == 1:
        print_usage()
        return

    command = sys.argv[1].strip()

    if command == 'info':
        info()
    elif command == 'code' and len(sys.argv) < 3:
        print_usage()
    elif command == 'code':
        code(*sys.argv[2:])
