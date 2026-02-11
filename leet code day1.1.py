# two sum
nums = [2,7,11,15]
target = 9
def twosum (nums,target):
    t={}
    for i,num in enumerate(nums):
        if target - num in t:
           return [t[target-num],i]
        t[num]=i
    return []

