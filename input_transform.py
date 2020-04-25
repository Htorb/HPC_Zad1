import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', help= 'paste path to biog.txt file')
args = parser.parse_args()

file_path = args.path

file_content = open(file_path).read().split('\n')

new_file_content =  [file_content[0]]

for line in file_content[1:]:
    if len(line) == 0:
        continue
    line = line.split(' ')
    new_file_content += [f"{line[0]} {line[1]} 1"]

new_file_content = '\n'.join(new_file_content)

with open(file_path + ".new", "w") as text_file:
    text_file.write(new_file_content + "\n")