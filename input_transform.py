import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', help= 'paste path to biog.txt file')
args = parser.parse_args()

file_path = args.path

file_content = open(file_path).read().split('\n')

first_line = True
new_file_content = []
for line in file_content:
    if len(line) == 0:
        continue
    if line[0] == '%':
        new_file_content += [line]
        continue
    line = line.split(' ')
    if first_line:
        new_file_content += [f"{line[0]} {line[1]} {line[2]}"]
        first_line = False
    else:
        new_file_content += [f"{line[0]} {line[1]} {abs(float(line[2]))}"]

new_file_content = '\n'.join(new_file_content)

with open(file_path + ".new", "w") as text_file:
    text_file.write(new_file_content + "\n")