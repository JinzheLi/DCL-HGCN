def countSelections(n, k, a, b, abilities, cooperations):
    left = right = 0
    count = 0
    while right < n:
        # 检查窗口是否为空
        if right - left + 1 < k:
            right += 1
            continue

        # 判断窗口内的能力值和合作值是否满足条件
        if min(abilities[left:right + 1]) >= a and min(cooperations[left:right + 1]) >= b:
            count += 1
            left += 1
        else:
            right += 1
            left = right
    return count

# 示例用法
n = 10
k = 4
a = 2
b = 4
abilities = [2,2,9,1,8,1,6,1,7,7]
cooperations = [4,8,5,1,9,4,1,3,9,4]
print(countSelections(n, k, a, b, abilities, cooperations))  # 输出为6
