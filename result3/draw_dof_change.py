import matplotlib.pyplot as plt
import numpy as np

# Data
continents = ['Europe', 'Asia', 'North America', 'South America', 'Africa', 'Australia']
dof_2010 = [28.11, 23.55, 17.83, 11.73, 10.39, 8.14]
dof_2015 = [30.46, 28.36, 18.12, 13.24, 12.40, 9.56]
dof_2020 = [32.43, 30.01, 18.39, 14.69, 13.38, 9.85]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Set width of bars
barWidth = 0.25

# Set positions of the bars on X axis
r1 = np.arange(len(continents))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create bars
bars1 = ax.bar(r1, dof_2010, width=barWidth, edgecolor='grey', label='2010', alpha=0.7, color='#4472C4')
bars2 = ax.bar(r2, dof_2015, width=barWidth, edgecolor='grey', label='2015', alpha=0.7, color='#5B9BD5')
bars3 = ax.bar(r3, dof_2020, width=barWidth, edgecolor='grey', label='2020', alpha=0.7, color='#70AD47')

# Add line chart overlay
# Calculate x positions for the line points (centered on each continent's group of bars)
line_x = [r + barWidth for r in r1]
ax2 = ax.twinx()  # Create a second y-axis sharing the same x-axis

# Plot the lines
ax2.plot(r1, dof_2010, 'o-', linewidth=2, color='#4472C4', label='2010 Trend')
ax2.plot(r2, dof_2015, 'o-', linewidth=2, color='#5B9BD5', label='2015 Trend')
ax2.plot(r3, dof_2020, 'o-', linewidth=2, color='#70AD47', label='2020 Trend')
ax2.set_ylim(0, max(dof_2020) * 1.1)  # Match the scale of the bar chart

# Add title and labels
ax.set_title('Continental River Fragmentation (DOF) Values: 2010-2020', fontsize=16, pad=20)
ax.set_xlabel('Continents', fontsize=14, labelpad=10)
ax.set_ylabel('DOF Value', fontsize=14)
ax2.set_ylabel('DOF Value (Line Trend)', fontsize=14)

# Add x-axis tick labels
ax.set_xticks([r + barWidth for r in range(len(continents))])
ax.set_xticklabels(continents, fontsize=12)

# Add value labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Create a combined legend for both the bar chart and line chart
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

# Add a grid for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()