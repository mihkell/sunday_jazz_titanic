s = 'aa ee rr rr aa xx cc'
s_array = s.split(' ')
s_set = set(s_array)
enumerated = enumerate(s_set)

print(list(enumerated))