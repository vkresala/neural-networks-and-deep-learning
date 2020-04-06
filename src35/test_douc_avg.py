from douc_avg import *

sorted_avgs = sorted_avgs(AVGS)


### test1
print("test0.05 > 1 --> ", guess_number(0.05, sorted_avgs))
print("test0.10 > 7 --> ", guess_number(0.1, sorted_avgs))
print("test0.15 > 8 --> ", guess_number(0.15, sorted_avgs))
print("test0.20 > 0 --> ", guess_number(0.2, sorted_avgs))