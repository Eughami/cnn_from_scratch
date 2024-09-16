import matplotlib.pyplot as plt
# 16	0.9389	554
# 32	0.9374	394
# 64	0.9358	314
# 128	0.9304	266

xx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
gpu =  [(i * 0.48 + 0.48) for i in range(20)]
cpu =  [(i * 30.5 + 30.5) for i in range(20)]
conv =  [(i * 19 + 19) for i in range(20)]
conv_back =  [(i * 3.3 + 3.3) for i in range(20)]
pool =  [(i * 27.55 + 27.55) for i in range(20)]
pool_back =  [(i * 11.8 + 11.8) for i in range(20)]
plt.figure(figsize=(8, 6))
# plt.xticks(xx, xx)
# Sequential CNN	610
# Parallel Convolution	380 
# Parallel Pooling	551 
# Parallel Convolution + Parallel Backpropagation	66 
# Parallel Pooling +  Parallel Backpropagation	236 
# Parallel CNN	9


plt.plot(xx, gpu, label="GPU", linestyle="--")
plt.plot(xx, cpu, label="CPU", linestyle=':')
plt.plot(xx, conv, label="Convolution", linestyle='-.')
plt.plot(xx, conv_back, label="Conv + Back", linestyle='-')
plt.plot(xx, pool, label="Pool", linestyle='-')
plt.plot(xx, pool_back, label="Pool + Back", linestyle='-')
# plt.plot(tt, at2, label='GPU')
# Adding labels and title
plt.xlabel('Number of Epochs')
plt.ylabel('Training time (min)')
# plt.title('Training time by batch size',color='white')

# Save the plot as an image (PNG format by default) with number of filter as parameter
plt.legend()
# plt.style.use("dark_background")
# plt.gca().set_facecolor('#121212')  # Dark gray background
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
# plt.tick_params(axis='x', colors='white')
# plt.tick_params(axis='y', colors='white')

# Darken the legend and set it to white
# plt.legend(facecolor='black', edgecolor='white', fontsize='small', labelcolor='white')
# Darken the plot border
# plt.gca().spines['top'].set_color('white')
# plt.gca().spines['bottom'].set_color('white')
# plt.gca().spines['left'].set_color('white')
# plt.gca().spines['right'].set_color('white')

# plt.show()
plt.savefig('different_parts_speedup_line_graph.png')
# plt.show()
# close the graph
plt.close()

