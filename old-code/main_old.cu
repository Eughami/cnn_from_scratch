#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>

#define CUDA_CHECK_ERROR(func)                                                    \
  do                                                                              \
  {                                                                               \
    cudaError_t err = func;                                                       \
    if (err != cudaSuccess)                                                       \
    {                                                                             \
      std::cerr << "\nCUDA error in " << #func << " at line " << __LINE__ << ": " \
                << cudaGetErrorString(err) << std::endl;                          \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (0)

// ! one by one convolution
// __global__ void convolutionLayerMultipleFilters(float *input, float *output, float *filters, float *biases,
//                                                 int inputWidth, int inputHeight, int numFilters, int filterSize)
// {
//   int outputWidth = inputWidth - filterSize + 1;
//   int outputHeight = inputHeight - filterSize + 1;

//   int outputX = blockIdx.x * blockDim.x + threadIdx.x;
//   int outputY = blockIdx.y * blockDim.y + threadIdx.y;
//   int filterIndex = blockIdx.z;

//   if (outputX < outputWidth && outputY < outputHeight && filterIndex < numFilters)
//   {
//     int inputXStart = outputX;
//     int inputYStart = outputY;
//     // printf("output: (%d,%d), inputXStart: %d, inputYStart:%d\n", outputX, outputY, inputXStart, inputYStart);

//     float result = 0.0f;
//     for (int i = 0; i < filterSize; ++i)
//     {
//       for (int j = 0; j < filterSize; ++j)
//       {
//         int currentX = inputXStart + i;
//         int currentY = inputYStart + j;

//         int inputIndex = currentX * inputWidth + currentY;
//         int filterIdx = (filterIndex * filterSize * filterSize) + (i * filterSize) + j;
//         // printf("(%d,%d,%d) result: %f + filterIdx:%d  %f x %f\n", outputX, outputY, filterIndex, result, filterIdx, input[inputIndex], filters[filterIdx]);
//         result += input[inputIndex] * filters[filterIdx];
//       }
//     }
//     // Add bias
//     // printf("(%d,%d) - result: %f , bias: %f\n", outputX, outputY, result, biases[filterIndex]);

//     // result += biases[filterIndex];

//     int outputIndex = (filterIndex * outputHeight + outputY) * outputWidth + outputX;
//     printf("(%d,%d,%d) - outputIndex: %d, result:%f\n", outputX, outputY, filterIndex, outputIndex, result);
//     output[outputIndex] = result;
//   }
// }

__global__ void convolutionLayerMultipleFilters(float *input, float *output, float *filters, float *biases,
                                                int inputWidth, int inputHeight, int numFilters, int filterSize, int numImages)
{
  int outputWidth = inputWidth - filterSize + 1;
  int outputHeight = inputHeight - filterSize + 1;

  int outputX = blockIdx.x * blockDim.x + threadIdx.x;
  int outputY = blockIdx.y * blockDim.y + threadIdx.y;
  int filterIndex = blockIdx.z;
  if (outputX < outputWidth && outputY < outputHeight && filterIndex < numFilters)
  {
    int imageIdx = threadIdx.z; // Each thread handles one image

    if (imageIdx < numImages)
    {
      int inputOffset = imageIdx * inputWidth * inputHeight;
      int outputOffset = imageIdx * numFilters * outputHeight * outputWidth;

      int inputXStart = outputX;
      int inputYStart = outputY;

      float result = 0.0f;

      for (int i = 0; i < filterSize; ++i)
      {
        for (int j = 0; j < filterSize; ++j)
        {
          int currentX = inputXStart + i;
          int currentY = inputYStart + j;

          int inputIndex = inputOffset + currentX * inputWidth + currentY;
          int filterIdx = (filterIndex * filterSize * filterSize) + (i * filterSize) + j;

          result += input[inputIndex] * filters[filterIdx];
        }
      }

      // Add bias
      // result += biases[filterIndex];

      int outputIndex = outputOffset + (filterIndex * outputHeight + outputY) * outputWidth + outputX;
      output[outputIndex] = result;
      printf("(%d,%d,%d) - outputIndex: %d, result:%f\n", outputX, outputY, filterIndex, outputIndex, result);
    }
  }
}

// __global__ void convolutionLayerMultipleFilters(float *input, float *output, float *filters, float *biases,
//                                                 int inputWidth, int inputHeight, int numFilters, int filterSize, int imageCount)
// {
//   int outputWidth = inputWidth - filterSize + 1;
//   int outputHeight = inputHeight - filterSize + 1;

//   int outputX = blockIdx.x * blockDim.x + threadIdx.x;
//   int outputY = blockIdx.y * blockDim.y + threadIdx.y;
//   int filterIndex = blockIdx.z;

//   int inputSize = inputWidth * inputHeight;
//   int outputSize = outputWidth * outputHeight;

//   for (int imageIdx = 0; imageIdx < imageCount; ++imageIdx)
//   {
//     int inputOffset = imageIdx * inputSize;
//     int outputOffset = imageIdx * numFilters * outputSize;

//     if (outputX < outputWidth && outputY < outputHeight && filterIndex < numFilters)
//     {
//       int inputStartX = outputX;
//       int inputStartY = outputY;

//       float result = 0.0f;

//       for (int i = 0; i < filterSize; ++i)
//       {
//         for (int j = 0; j < filterSize; ++j)
//         {
//           int inputX = inputStartX + i;
//           int inputY = inputStartY + j;

//           int inputIndex = inputOffset + inputX * inputWidth + inputY;
//           int filterIdx = filterIndex * filterSize * filterSize + i * filterSize + j;

//           result += input[inputIndex] * filters[filterIdx];
//         }
//       }

//       // Add bias
//       result += biases[filterIndex];

//       int outputIndex = outputOffset + filterIndex * outputSize + outputY * outputWidth + outputX;
//       output[outputIndex] = result;
//       printf("(%d,%d,%d) - outputIndex: %d, result:%f\n", outputX, outputY, filterIndex, outputIndex, result);
//     }
//   }
// }

// Define the CUDA kernel for the max pooling operation
__global__ void maxPoolingLayer(float *input, float *output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
  // int outputX = blockIdx.x * blockDim.x + threadIdx.x;
  // int outputY = blockIdx.y * blockDim.y + threadIdx.y;

  // if (outputX < outputWidth && outputY < outputHeight)
  // {
  //   int inputX = outputX * 2; // Assuming 2x2 max pooling
  //   int inputY = outputY * 2;

  //   float maxVal = -FLT_MAX; // Initialize with a very small value

  //   // Perform max pooling
  //   for (int i = 0; i < 2; ++i)
  //   {
  //     for (int j = 0; j < 2; ++j)
  //     {
  //       int currentX = inputX + i;
  //       int currentY = inputY + j;

  //       // Boundary checking
  //       if (currentX < inputWidth && currentY < inputHeight)
  //       {
  //         float value = input[currentY * inputWidth + currentX];
  //         maxVal = fmaxf(maxVal, value);
  //       }
  //     }
  //   }

  //   // Store the max value in the output array
  //   output[outputY * outputWidth + outputX] = maxVal;
  // }
}

// Define the CUDA kernel for the dense layer operation
__global__ void denseLayer(float *input, float *output, int inputSize, int outputSize)
{
  // int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // if (tid < outputSize)
  // {
  //   float sum = 0.0f;
  //   for (int i = 0; i < inputSize; ++i)
  //   {
  //     // Simple dense layer operation: multiply input values by weights and accumulate
  //     sum += input[i] * /* Add your weights here */;
  //   }

  //   // Apply activation function (e.g., ReLU)
  //   output[tid] = fmaxf(0.0f, sum); // ReLU activation function
  // }
}

// Define the CUDA kernel for the output layer operation
__global__ void outputLayer(float *input, float *output, int inputSize, int outputSize)
{
  // int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // if (tid < outputSize)
  // {
  //   // Apply softmax activation to get class probabilities
  //   float expSum = 0.0f;
  //   for (int i = 0; i < inputSize; ++i)
  //   {
  //     expSum += expf(input[i]);
  //   }

  //   // Calculate softmax for each class
  //   output[tid] = expf(input[tid]) / expSum;
  // }
}

bool loadMNISTData(const std::string &filename, std::vector<std::vector<float>> &images, std::vector<int> &labels)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "Unable to open file " << filename << std::endl;
    return false;
  }

  std::string line;
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::vector<float> row;
    std::string pixel;

    // Extract the label (first value in each row)
    std::getline(ss, pixel, ',');
    labels.push_back(std::stoi(pixel));

    // Extract pixel values
    while (std::getline(ss, pixel, ','))
    {
      row.push_back(std::stof(pixel) / 255.0f);
    }

    images.push_back(row);
  }

  file.close();
  return true;
}

// Placeholder function for training the model
void trainModel(/* parameters */)
{
  // Implement model training here
}

// Placeholder function for testing the model
void testModel(/* parameters */)
{
  // Implement model testing here
}

// Placeholder function for evaluating the model
void evaluateModel(/* parameters */)
{
  // Implement model evaluation here
}

// This function displays the image in the terminal
void displayImage(const std::vector<float> &image, int width, int height)
{
  for (int i = 0; i < height; ++i)
  {
    for (int j = 0; j < width; ++j)
    {
      // Convert pixel value to ASCII character based on intensity
      int pixel = static_cast<int>(image[i * width + j] * 9.0f);
      char character = pixel > 0 ? ('0' + pixel) : ' ';

      std::cout << character << ' ';
    }
    std::cout << std::endl;
  }
}

void initializeAndFlattenFilters(std::vector<std::vector<std::vector<float>>> &filters,
                                 std::vector<float> &flattenedFilters,
                                 int numFilters, int filterSize)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  filters.resize(numFilters, std::vector<std::vector<float>>(filterSize, std::vector<float>(filterSize)));

  for (int f = 0; f < numFilters; ++f)
  {
    for (int i = 0; i < filterSize; ++i)
    {
      for (int j = 0; j < filterSize; ++j)
      {
        float randValue = dis(gen);
        float roundedValue = roundf(randValue * 100.0f) / 100.0f; // Round to two decimal places
        filters[f][i][j] = roundedValue;
        flattenedFilters.push_back(filters[f][i][j]); // Append the value to the flattened vector
      }
    }
  }
}

// Function to flatten a vector of vectors
template <typename T>
std::vector<T> flattenVector(const std::vector<std::vector<T>> &input)
{
  std::vector<T> flattened;

  for (const auto &innerVec : input)
  {
    flattened.insert(flattened.end(), innerVec.begin(), innerVec.end());
  }

  return flattened;
}

// Function to initialize biases with random values
void initializeBiases(std::vector<float> &biases, int numFilters)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dis(0.0f, 0.1f); // Adjust the mean and standard deviation as needed

  biases.resize(numFilters);

  for (int f = 0; f < numFilters; ++f)
  {
    biases[f] = dis(gen);
  }
}

int main()
{
  // Define input data dimensions
  int inputWidth = 10;
  int inputHeight = 10;
  // int inputChannels = 1;

  // Define convolution parameters
  int filterSize = 3;
  int numFiltersConv1 = 3;
  // int numFiltersConv2 = 64;

  // // Define max pooling parameters
  // int poolSize = 2;

  // // Define dense layer parameters
  // int denseInputSize = 7 * 7 * numFiltersConv2; // Based on the output size after convolutions and pooling
  // int denseOutputSize = 128;

  // // Define output layer parameters
  // // int outputSize = 10; // Number of classes

  // // Allocate memory for layers
  // float *input_data;
  // float *conv1_output;
  // float *pool1_output;
  // float *conv2_output;
  // float *pool2_output;
  // float *dense_output;

  // cudaMalloc(&input_data, inputWidth * inputHeight * inputChannels * sizeof(float));
  // cudaMalloc(&conv1_output, /* Calculate size */);
  // cudaMalloc(&pool1_output, /* Calculate size */);
  // cudaMalloc(&conv2_output, /* Calculate size */);
  // cudaMalloc(&pool2_output, /* Calculate size */);
  // cudaMalloc(&dense_output, denseOutputSize * sizeof(float));
  // cudaMalloc(&output, outputSize * sizeof(float));

  // // Load data
  std::vector<std::vector<float>> mnistImages;
  std::vector<int> mnistLabels;
  std::string filename = "mnist_test.csv"; // Update with your file path
  if (loadMNISTData(filename, mnistImages, mnistLabels))
  {
    const int imageWidth = 28;  // MNIST image width
    const int imageHeight = 28; // MNIST image height

    // std::cout << "Label of the first image: " << mnistLabels[0] << std::endl;
    // std::cout << "Displaying the first image:" << std::endl;
    // if (!mnistImages.empty() && !mnistImages[0].empty())
    // {
    //   displayImage(mnistImages[0], imageWidth, imageHeight);
    // }
    // else
    // {
    //   std::cout << "No data available." << std::endl;
    // }
  }

  else
  {
    std::cout << "Failed to load MNIST data" << std::endl;
  }

  std::vector<std::vector<std::vector<float>>> filters;
  std::vector<float> flattenedFilters;

  std::vector<std::vector<std::vector<float>>> input;
  std::vector<float> flattenedInput;
  std::vector<float> biases;
  int imagesToShow = 5; // Change this value to display output for a different number of images

  // ! Clean all the mess , we should only have the loading, convolution and maybe a pooling
  // TODO. Add the CPU version for comparison
  initializeAndFlattenFilters(filters, flattenedFilters, numFiltersConv1, filterSize);
  initializeAndFlattenFilters(input, flattenedInput, imagesToShow, inputWidth);
  initializeBiases(biases, numFiltersConv1);
  // Flatten the vector of vectors
  // print mnistImages size
  std::cout << "mnistImages size: " << mnistImages.size() << std::endl;
  std::vector<float> mnistImagesFlattened = flattenVector(mnistImages);
  // print mnistImagesFlattened size
  // std::cout << "mnistImagesFlattened size: " << mnistImagesFlattened.size() << std::endl;
  // // print mnistImagesFlattened til index 784
  // std::cout << "\nmnistImagesFlattened til index 784\n"
  //           << std::endl;
  // for (int i = 0; i < 784; ++i)
  // {
  //   std::cout << mnistImagesFlattened[i] << " ";
  // }
  // // print mnsitimages index 0 as a single line
  // std::cout << "\nmnistImages index 0 as a single line\n"
  //           << std::endl;
  // for (int i = 0; i < 784; ++i)
  // {
  //   std::cout << mnistImages[0][i] << " ";
  // }

  std::cout << "Flattened input size : " << flattenedInput.size() << std::endl;

  for (int i = 0; i < imagesToShow; ++i)
  {
    std::cout << "Image " << i << std::endl;
    for (int j = 0; j < 5; ++j)
    {
      for (int k = 0; k < 5; ++k)
      {
        std::cout << input[i][j][k] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  // // // print filters
  for (int i = 0; i < numFiltersConv1; ++i)
  {
    std::cout << "Filter " << i << std::endl;
    for (int j = 0; j < filterSize; ++j)
    {
      for (int k = 0; k < filterSize; ++k)
      {
        std::cout << filters[i][j][k] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // // print flattened filters
  // std::cout << "Flattened filters" << std::endl;
  // for (int i = 0; i < flattenedFilters.size(); ++i)
  // {
  //   std::cout << flattenedFilters[i] << " ";
  // }

  // // print biases
  // std::cout << "Biases" << std::endl;
  // for (int i = 0; i < numFiltersConv1; ++i)
  // {
  //   std::cout << biases[i] << " ";
  // }

  // // print first image in mnistImagesFlattened
  // std::cout << "First image in mnistImagesFlattened" << std::endl;
  // for (int i = 0; i < 28; ++i)
  // {
  //   for (int j = 0; j < 28; ++j)
  //   {
  //     std::cout << mnistImagesFlattened[i * 28 + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // int imagesToShow = mnistImages.size(); // Change this value to display output for a different number of images

  float *d_input, *d_output, *d_filters, *d_biases;
  size_t filterMemSize = flattenedFilters.size() * sizeof(float);
  size_t biasSize = biases.size() * sizeof(float);
  size_t inputSize = flattenedInput.size() * sizeof(float);
  int height = inputHeight - filterSize + 1;
  int width = inputWidth - filterSize + 1;
  size_t outputSize = imagesToShow * width * height * numFiltersConv1 * sizeof(float);
  float *output = new float[outputSize];
  memset(output, 0, outputSize);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // // print output
  // std::cout << "Output" << std::endl;
  for (int i = 0; i < numFiltersConv1; ++i)
  {
    std::cout << "Filter " << i << std::endl;
    for (int j = 0; j < height; ++j)
    {
      for (int k = 0; k < width; ++k)
      {
        std::cout << output[i * width * height + j * width + k] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "\nInput size: " << inputSize << std::endl;
  // print output size
  std::cout << "\nOutput size: " << outputSize << std::endl;
  std::cout << "\nfilterMemSize size: " << filterMemSize << std::endl;
  std::cout << "\nbiasSize size: " << biasSize << std::endl;
  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  std::cout << "Free memory: " << freeMem << " bytes\n";
  std::cout << "Total memory: " << totalMem << " bytes\n";
  CUDA_CHECK_ERROR(cudaMalloc(&d_input, inputSize));
  CUDA_CHECK_ERROR(cudaMalloc(&d_output, outputSize));
  CUDA_CHECK_ERROR(cudaMalloc(&d_filters, filterMemSize));
  CUDA_CHECK_ERROR(cudaMalloc(&d_biases, biasSize));

  // print alloc success
  std::cout << "\nAlloc success" << std::endl;
  // copy input data to device
  CUDA_CHECK_ERROR(cudaMemcpy(d_input, flattenedInput.data(), inputSize, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_filters, flattenedFilters.data(), filterMemSize, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_biases, biases.data(), biasSize, cudaMemcpyHostToDevice));
  // print copy success
  std::cout << "\nCopy success" << std::endl;

  // Call the CUDA kernel for the convolution layer
  dim3 threadsPerBlock(3, 3, imagesToShow);
  // int threadsPerBlock = 256;

  int numBlocksX = (inputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x;
  int numBlocksY = (inputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y;
  std::cout << "numBlocksX: " << numBlocksX << ", numBlocksY: " << numBlocksY << std::endl;
  dim3 numBlocks(numBlocksX, numBlocksY, numFiltersConv1);

  // ? this are manual settings
  // Assuming each image is 28x28 and there are numImages images in total.
  // const int imageSize = inputWidth * inputHeight;
  // const int totalInputSize = imagesToShow * imageSize;
  // int numBlocks = (totalInputSize + threadsPerBlock - 1) / threadsPerBlock;

  cudaEventRecord(start);

  std::cout << "numBlocksX: " << numBlocksX << ", numBlocksY: " << numBlocksY << ", numBlocksZ: " << numFiltersConv1 << std::endl;
  convolutionLayerMultipleFilters<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_filters, d_biases,
                                                                  inputWidth, inputHeight, numFiltersConv1, filterSize, imagesToShow);
  // convolutionLayerMultipleFilters<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_filters, d_biases,
  //                                                                 inputWidth, inputHeight, numFiltersConv1, filterSize, imagesToShow);
  cudaDeviceSynchronize();
  // Record stop event
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  std::cout << "\nConvolution success" << std::endl;
  std::cout << "\nDevice sync success" << std::endl;
  // Copy results from device to host
  CUDA_CHECK_ERROR(cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost));

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

  // Cleanup events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // ? my filter
  std::cout << "\nOutput size: " << outputSize << std::endl;
  for (int i = 0; i < imagesToShow * numFiltersConv1 * width * height; ++i)
  {
    if (i % (numFiltersConv1 * width * height) == 0)
      std::cout << "\n\nImage " << i / (numFiltersConv1 * width * height) << std::endl;
    if (i % 9 == 0)
      std::cout << "\n\nFilter " << (i / 9) % 3;
    if (i % 3 == 0)
      std::cout << std::endl;
    std::cout << output[i] << " ";
  }

  // free memory
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_filters);
  cudaFree(d_biases);

  // // Convolutional Layer 1
  // convolutionLayer<<<...>>>(input_data, conv1_output, inputWidth, inputHeight, /* Pass output dimensions */);

  // // Max Pooling Layer 1
  // maxPoolingLayer<<<...>>>(conv1_output, pool1_output, /* Pass dimensions */);

  // // Convolutional Layer 2
  // convolutionLayer<<<...>>>(pool1_output, conv2_output, /* Pass dimensions */);

  // // Max Pooling Layer 2
  // maxPoolingLayer<<<...>>>(conv2_output, pool2_output, /* Pass dimensions */);

  // // Dense Layer
  // denseLayer<<<...>>>(pool2_output, dense_output, denseInputSize, denseOutputSize);

  // // Output Layer
  // outputLayer<<<...>>>(dense_output, output, denseOutputSize, outputSize);

  // // Training the model
  // trainModel(/* Pass training parameters */);

  // // Testing the model
  // testModel(/* Pass testing parameters */);

  // // Evaluate the model
  // evaluateModel(/* Pass evaluation parameters */);

  // // Clean up: deallocate memory
  // cudaFree(input_data);
  // cudaFree(conv1_output);
  // cudaFree(pool1_output);
  // cudaFree(conv2_output);
  // cudaFree(pool2_output);
  // cudaFree(dense_output);
  // cudaFree(output);

  return 0;
}
