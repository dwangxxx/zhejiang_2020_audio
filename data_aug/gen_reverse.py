lines = open('audio_2.txt', 'r').readlines()
des = open('audio_2_r.txt', 'w')

for i in range(19999, 0, -1):
    des.write(lines[i])