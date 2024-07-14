To conduct these experiments the following hyperparameters were used :
Learning rate: 0.01
Batch size: 16
Number of epochs: 20
Number of filters: 8

Accuracy
Training Time (in minutes)
CPU (MNIST)
0.9463
609.2
Tensorflow
405.73 seconds
97.12%
443.10 seconds
97.80%

GPU (MNIST)
0.9495
8.6 (98.58 % faster)
Tensorflow
229.30 seconds
265.03 seconds
262.89 seconds

0.9711

CPU (FASHION_MNIST)
0.8358
615
Tensorflow
443.55 seconds
87.52%
86.91%

GPU (FASHION_MNIST)
0.8330
9.1 (98.52 % faster)
Tensorflow
265.03 seconds
262.89 seconds
262.98 seconds

0.8624

Tensorflow is faster and slightly more accurate, this is expected of the state of the art solution
for CNN models.
Furthermore when compared to Tensorflow training time for the GPU version is about 30% slower which is acceptable when compared to the state of the art solution right now.Will be moved to the related work comparison table below.

2 conv + 2pool time
MNIST
537 seconds
0.9302
FASHION_MNIST
533 seconds
83.50%

The weights update being used in this study is a very simple gradient descend that
average all filters gradients for the batch. This works fine for simple models but a more well tuned
update method is prefered specially for deeper models. As shown for the double layer result accuracy is
basically the same for both datasets, thus showing that with a better update algorithm or optimizer such as Adam
the accuracy would have been higher. In this study we mainly focus on parallelizing the operations so the fact that
a model with twice as many operations basically completes training within the same timeframe shows that the CUDA optimized
operations are actually working.
