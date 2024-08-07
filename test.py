times = int(input('How many times should I repeat this line?\nInput: '))
counts = []
for _ in range(times):
    count = int(input('Input: '))
    if len(counts) < 100 and count < 300:
        counts.append(count)
    else:
        break
summ = 0
for i in counts:
    if i % 6 == 0:
        summ += i
print(summ)