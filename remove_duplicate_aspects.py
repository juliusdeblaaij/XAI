import re


def remove_duplicate_aspects(lst):
    sorted_lst = list(sorted(lst, key=len))

    filtered_lst = []
    for i in range(len(sorted_lst)):
        is_common_start = False
        for j in range(i):

            if re.match("^" + sorted_lst[j], sorted_lst[i]) != None:
                is_common_start = True
                break
        if not is_common_start:
            filtered_lst.append(sorted_lst[i])
    return filtered_lst
