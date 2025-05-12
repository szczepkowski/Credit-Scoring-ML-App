x = 256
total = 0
while x > 0:
 if total > 500:
   break
 total += x
 x = x // 2
 print(x)