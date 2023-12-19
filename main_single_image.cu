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

// !one by one convolution
__global__ void
convolutionLayerMultipleFilters(float *input, float *output, float *filters,
                                int inputWidth, int inputHeight, int numFilters, int filterSize)
{
  int outputWidth = inputWidth - filterSize + 1;
  int outputHeight = inputHeight - filterSize + 1;

  int outputX = blockIdx.x * blockDim.x + threadIdx.x;
  int outputY = blockIdx.y * blockDim.y + threadIdx.y;
  int filterIndex = blockIdx.z;

  if (outputX < outputWidth && outputY < outputHeight && filterIndex < numFilters)
  {
    int inputXStart = outputX;
    int inputYStart = outputY;
    // printf("output: (%d,%d), inputXStart: %d, inputYStart:%d\n", outputX, outputY, inputXStart, inputYStart);

    float result = 0.0f;
    for (int i = 0; i < filterSize; ++i)
    {
      for (int j = 0; j < filterSize; ++j)
      {
        int currentX = inputXStart + i;
        int currentY = inputYStart + j;

        int inputIndex = currentX * inputWidth + currentY;
        int filterIdx = (filterIndex * filterSize * filterSize) + (i * filterSize) + j;
        // printf("(%d,%d,%d) result: %f + filterIdx:%d  %f x %f\n", outputX, outputY, filterIndex, result, filterIdx, input[inputIndex], filters[filterIdx]);
        result += input[inputIndex] * filters[filterIdx];
      }
    }

    int outputIndex = (filterIndex * outputHeight + outputY) * outputWidth + outputX;
    printf("(%d,%d,%d) - outputIndex: %d, result:%f\n", outputX, outputY, filterIndex, outputIndex, result);
    output[outputIndex] = result;
  }
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

void displayArray(std::vector<std::vector<float>> &array, int width) // assuming width=height
{
  std::cout << std::endl;
  for (int i = 0; i < width; ++i)
  {
    for (int j = 0; j < width; ++j)
    {
      std::cout << array[i][j] << " ";
    }
    std::cout << "\n";
  }
}

int main()
{
  int inputWidth = 28;
  int inputHeight = 28;

  int filterSize = 3;      // 3x3 filter
  int numFiltersConv1 = 5; // 3 filters

  int imagesToShow = 10; // Change this value to display output for a different number of images

  // Load data
  // std::vector<std::vector<float>> mnistImages;
  // std::vector<int> mnistLabels;
  // std::string filename = "mnist_test.csv"; // Update with your file path
  // if (!loadMNISTData(filename, mnistImages, mnistLabels))
  // {
  //   exit(EXIT_FAILURE);
  // }

  // std::vector<float> mnistImagesFlattened = flattenVector(mnistImages);

  std::vector<std::vector<std::vector<float>>> filters;
  std::vector<float> flattenedFilters;
  initializeAndFlattenFilters(filters, flattenedFilters, numFiltersConv1, filterSize);

  std::vector<std::vector<std::vector<float>>> input;
  std::vector<float> flattenedInput;
  initializeAndFlattenFilters(input, flattenedInput, imagesToShow, inputWidth);

  for (int i = 0; i < imagesToShow; i++)
  {
    std::cout << "\nImage " << i << std::endl;
    displayArray(input[i], inputWidth);
  }

  for (int i = 0; i < numFiltersConv1; i++)
  {
    std::cout << "\nFilter " << i << std::endl;
    displayArray(filters[i], filterSize);
  }

  float *d_input, *d_output, *d_filters;
  size_t filterMemSize = flattenedFilters.size() * sizeof(float);
  size_t inputSize = flattenedInput.size() * sizeof(float);
  int height = inputHeight - filterSize + 1;
  int width = inputWidth - filterSize + 1;
  size_t outputSize = imagesToShow * width * height * numFiltersConv1 * sizeof(float);
  float *output = new float[outputSize];
  memset(output, 0, outputSize);

  std::cout << "\nInput size: " << inputSize << std::endl;
  std::cout << "\nOutput size: " << outputSize << std::endl;
  std::cout << "\nfilterMemSize size: " << filterMemSize << std::endl;

  CUDA_CHECK_ERROR(cudaMalloc(&d_input, inputSize));
  CUDA_CHECK_ERROR(cudaMalloc(&d_output, outputSize));
  CUDA_CHECK_ERROR(cudaMalloc(&d_filters, filterMemSize));
  std::cout << "\nAlloc success" << std::endl;

  CUDA_CHECK_ERROR(cudaMemcpy(d_input, flattenedInput.data(), inputSize, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_filters, flattenedFilters.data(), filterMemSize, cudaMemcpyHostToDevice));
  std::cout << "\nMemcpy success" << std::endl;

  // Call the CUDA kernel for the convolution layer
  dim3 blockSize(16, 16);
  // int threadsPerBlock = 256;

  int numBlocksX = (inputWidth + blockSize.x - 1) / blockSize.x;
  int numBlocksY = (inputHeight + blockSize.y - 1) / blockSize.y;
  std::cout << "numBlocksX: " << numBlocksX << ", numBlocksY: " << numBlocksY << std::endl;
  dim3 numBlocks(numBlocksX, numBlocksY, numFiltersConv1);

  convolutionLayerMultipleFilters<<<numBlocks, blockSize>>>(d_input, d_output, d_filters,
                                                            inputWidth, inputHeight,
                                                            numFiltersConv1, filterSize);

  CUDA_CHECK_ERROR(cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost));

  std::cout << "\nOutput size: " << outputSize << std::endl;
  int singleFilter = width * height;
  for (int i = 0; i < imagesToShow * numFiltersConv1 * width * height; ++i)
  {
    if (i % (numFiltersConv1 * width * height) == 0)
      std::cout << "\n\nImage " << i / (numFiltersConv1 * width * height) << std::endl;
    if (i % (singleFilter) == 0)
      std::cout << "\n\nFilter " << (i / singleFilter) % width;
    if (i % width == 0)
      std::cout << std::endl;
    std::cout << output[i] << " ";
  }

  CUDA_CHECK_ERROR(cudaFree(d_input));
  CUDA_CHECK_ERROR(cudaFree(d_output));
  CUDA_CHECK_ERROR(cudaFree(d_filters));

  return 0;
}