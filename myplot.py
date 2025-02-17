# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
''' 
name_list = ['codeine','fentanyl','hydrocodone','norco','oxycodone','oxycontin','percocet','vicodin']
num_list = [11581,20783,466,1078,1053,3558,4351,1600]
plt.barh(range(len(num_list)), num_list,tick_label = name_list)
plt.ylabel("kind of opioid")
plt.xlabel("number of related tweets")
plt.show()'''

name_list = ['codeine','fentanyl','oxycontin','percocet']
num_list = [10679,10649,1951,2640]
plt.barh(range(len(num_list)), num_list,tick_label = name_list)
plt.ylabel("kind of opioid")
plt.xlabel("number of related tweets(no RT)")
plt.show()

name_list = ['codeine','fentanyl','oxycontin','percocet']
num_list = [22126,32839,5151,6454]
plt.barh(range(len(num_list)), num_list,tick_label = name_list)
plt.ylabel("kind of opioid")
plt.xlabel("number of related tweets")
plt.show()