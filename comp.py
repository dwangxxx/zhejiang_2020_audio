result1 = open('result/result_round2.txt').readlines()
result2 = open('result/result.txt').readlines()

for i in result2:
    if i not in result1:
        print(i)