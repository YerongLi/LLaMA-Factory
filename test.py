import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")
#    total_bill   tip     sex smoker  day    time  size
# 0       16.99  1.01  Female     No  Sun  Dinner     2
# 1       10.34  1.66    Male     No  Sun  Dinner     3
# 2       21.01  3.50    Male     No  Sun  Dinner     3
# 3       23.68  3.31    Male     No  Sun  Dinner     2
# 4       24.59  3.61  Female     No  Sun  Dinner     4

hatches = ['\\\\', '//']

fig, ax = plt.subplots(figsize=(6,3))
print(tips.head())


sns.barplot(data=tips, x="day", y="total_bill", hue="time",linewidth=2)

# loop through days
for hues, hatch in zip(ax.containers, hatches):
    # set a different hatch for each time
    for hue in hues:
        hue.set_hatch(hatch)

# add legend with hatches
plt.legend().loc='best'

plt.show()