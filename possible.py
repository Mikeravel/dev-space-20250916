import initial as ini
# mn = [11,2,7,16,20,5,21,4,8,12,14,1,24,17,13,18,10,23,0,9,19,3,15,22,6]
# un = [7,7,8,7,3,3,0,3,9,3,0,2,3,2,4,2,7,5,0,1,0,6,1,2,8]

def factorial(n):
    result = 1
    for i in range(1,n+1):
        result = result * i
    if result > 200:
        result = 200
    return result

