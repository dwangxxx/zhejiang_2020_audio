import os

all_file = open('audio_2.txt', 'r').readlines()
part_file = open('audio_50.txt', 'r').readlines()
add_file = open('audio_add.txt', 'w')

for line in all_file:
    if line.split(' ')[1].strip() == '1':
        add_file.write(line)

for line in part_file:
    add_file.write(line)