import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

hatches = ['\\\\', '//']

fig, ax = plt.subplots(figsize=(6,3))
print(tips.head())
sns.barplot(data=tips, x="day", y="total_bill", hue="time")

# loop through days
for hues, hatch in zip(ax.containers, hatches):
    # set a different hatch for each time
    for hue in hues:
        hue.set_hatch(hatch)

# add legend with hatches
plt.legend().loc='best'

plt.show()