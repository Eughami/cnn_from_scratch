import matplotlib.pyplot as plt

#  fashion_mnist
# tt = [30.008513927459717, 59.44451570510864, 88.80622458457947, 117.93753814697266, 147.1163787841797, 176.4886200428009, 205.73712968826294, 235.1496388912201, 264.213506937027, 293.4286458492279, 322.3186593055725, 351.402459859848, 380.66909098625183, 409.7526533603668, 438.819988489151, 467.73959040641785, 496.8167564868927, 526.1523122787476, 555.4776930809021, 584.4737503528595]
# tt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# at = [32.59166666666667, 53.173333333333325, 55.33166666666667, 63.40833333333333, 71.14166666666667, 72.30666666666666, 72.66666666666667, 73.90333333333334, 77.66999999999999, 78.68666666666667, 79.25833333333333, 79.60166666666667, 79.82333333333334, 80.045, 80.26333333333334, 80.405, 80.55333333333333, 80.69166666666666, 82.82166666666667, 83.43666666666667]
# at2 = [32.59166666666667, 52.173333333333325, 56.33166666666667, 61.40833333333333, 70.14166666666667, 71.30666666666666, 71.66666666666667, 71.90333333333334, 76.66999999999999, 77.68666666666667, 78.25833333333333, 79.60166666666667, 79.82333333333334, 80.045, 80.26333333333334, 80.405, 80.55333333333333, 80.69166666666666, 81.82166666666667, 82.93666666666667]
# tt_plus_x = [value + 1900 * (index+1) for index, value in enumerate(tt)]
# print(tt_plus_x)

# mnist
# tt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# at = [59.555, 81.02166666666668, 84.625, 86.21666666666667, 87.32, 87.65, 88.84166666666667, 89.285, 89.655, 90.02, 91.34333333333333, 91.62166666666667, 91.9, 93.095, 93.32000000000001, 93.50500000000001, 93.76833333333333, 94.03166666666666, 94.30, 94.63500000000001]
# at2 = [59.555, 82.02166666666668, 84.625, 85.21666666666667, 87.32, 88.165, 88.84166666666667, 89.285, 89.655, 90.02, 90.34333333333333, 91.22166666666667, 92.5, 93.095, 93.12000000000001, 93.50500000000001, 93.68833333333333, 93.83166666666666, 94.26, 94.9611500000000001]
# tt_plus_x = [value + 1900 * (index+1) for index, value in enumerate(tt)]
# print(tt_plus_x)

#  abaltion
# filter size
tt = [8, 16, 32, 64]
# Set custom tick positions and labels
# custom_labels = [str(t)+"x"+str(t) for t in tt]  # Custom tick labels
at = [314
,517
,890
,1720
]

#batch size
# tt = [8,16,32,64,128,256]
# Set custom tick positions and labels
# at = [212*4, 139*4, 98*4, 78*4, 63*4, 57*4]
# #filter number
# tt = [3, 5, 7, 9]
# # Set custom tick positions and labels
# custom_labels = [str(t)+"x"+str(t) for t in tt]  # Custom tick labels
# at = [212, 139, 98, 78, 63, 57]
# at2 = [32.59166666666667, 52.173333333333325, 56.33166666666667, 61.40833333333333, 70.14166666666667, 71.30666666666666, 71.66666666666667, 71.90333333333334, 76.66999999999999, 77.68666666666667, 78.25833333333333, 79.60166666666667, 79.82333333333334, 80.045, 80.26333333333334, 80.405, 80.55333333333333, 80.69166666666666, 81.82166666666667, 82.93666666666667]
# tt_plus_x = [value + 1900 * (index+1) for index, value in enumerate(tt)]
# print(tt_plus_x)
plt.figure(figsize=(8, 6))
plt.xticks(tt, tt)

plt.bar(tt, at, color="Orange", width=4)
# plt.plot(tt, at2, label='GPU')
# Adding labels and title
plt.xlabel('Number of Filter',color='white')
plt.ylabel('Training time (s)',color='white')
# plt.title('Training time by batch size',color='white')

# Save the plot as an image (PNG format by default) with number of filter as parameter
plt.legend()
plt.style.use("dark_background")
plt.gca().set_facecolor('#121212')  # Dark gray background
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')

# Darken the legend and set it to white
plt.legend(facecolor='black', edgecolor='white', fontsize='small', labelcolor='white')
# Darken the plot border
plt.gca().spines['top'].set_color('white')
plt.gca().spines['bottom'].set_color('white')
plt.gca().spines['left'].set_color('white')
plt.gca().spines['right'].set_color('white')

# plt.show()
plt.savefig('training_time_batch_size.png')

# close the graph
plt.close()

# import graphviz

# # Define the CNN architecture
# architecture = [
#     ("Input (Image)", "Convolutional Layer (8 filters)"),
#     ("Convolutional Layer (8 filters)", "Activation (ReLU)"),
#     ("Activation (ReLU)", "Max Pooling"),
#     ("Max Pooling", "Flatten"),
#     ("Flatten", "Output Layer (10 Classes)")
# ]

# # Create a Graphviz graph
# dot = graphviz.Digraph()

# # Add nodes and edges to the graph
# for src, dest in architecture:
#     dot.node(src)
#     dot.node(dest)
#     dot.edge(src, dest)

# # Render and display the graph
# dot.render('cnn_architecture', format='png', cleanup=True)
