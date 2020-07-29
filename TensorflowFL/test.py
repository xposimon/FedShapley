from scipy.special import comb
from itertools import permutations

def PowerSetsBinary(items):
    N = len(items)
    set_all = []
    for i in range(2 ** N):
        combo = []
        for j in range(N):
            if (i >> j) % 2 == 1:
                combo.append(items[j])
        set_all.append(combo)
    return set_all

def shapley_list_indexed(original_l, ll):
    for i in range(len(ll)):
        if set(ll[i]) == set(original_l):
            return i
    return -1

def remove_list_indexed(removed_ele, original_l, ll):
    new_original_l = []
    for i in original_l:
        new_original_l.append(i)
    for i in new_original_l:
        if i == removed_ele:
            new_original_l.remove(i)
    for i in range(len(ll)):
        if set(ll[i]) == set(new_original_l):
            return i
    return -1



group_shapley_value = [	0.23934978
,	0.3871076
,	0.37970853
,	0.3837444
,	0.38419282
,	0.38576233
,	0.38273543
,	0.38464126
,	0.38576233
,	0.38632286
,	0.38340807
,	0.3850897
,	0.38520178
,	0.38565022
,	0.3840807
,	0.38486546
,	0.3820628
,	0.38452914
,	0.38127804
,	0.3837444
,	0.38363227
,	0.38497758
,	0.38295963
,	0.38396862
,	0.3840807
,	0.3850897
,	0.38329595
,	0.3850897
,	0.38452914
,	0.3853139
,	0.38452914
,	0.38475338]

# print(remove_list_index, shapley_list_indexed(j, all_sets), group_shapley_value[shapley_list_indexed(j, all_sets)])
NUM_AGENT = 5
all_sets = PowerSetsBinary([i for i in range(NUM_AGENT)])

print(len(group_shapley_value))

s=sorted([i for i in range(NUM_AGENT)])
l=permutations(s)
all_orders = []
for x in l:
    all_orders.append(list(x))

agent_shapley = []
for index in range(NUM_AGENT):
    shapley = 0.0
    for order in all_orders:
        pos = order.index(index)
        pre_list = list(order[:pos])
        edge_list = list(order[:pos+1])
        pre_list_index = remove_list_indexed(index, pre_list, all_sets)
        #print(order ,pre_list, pre_list_index, all_sets[pre_list_index])

        if pre_list_index != -1:
            #print(j, remove_list_index, all_sets[remove_list_index])
            shapley += (group_shapley_value[shapley_list_indexed(edge_list, all_sets)] - group_shapley_value[
                pre_list_index]) / len(all_orders)
    agent_shapley.append(shapley)

for ag_s in agent_shapley:
    print(ag_s)

print(sum(agent_shapley), group_shapley_value[0])

#print(all_sets)