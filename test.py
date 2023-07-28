from datetime import datetime as dt 

a = dt.now()
b = a.strftime("%y%m%d")

print(a)
print(b)