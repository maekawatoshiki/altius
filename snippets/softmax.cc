#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void softmax(const std::vector<float> &input, std::vector<float> &output) {
  assert(input.size() > 0);
  assert(output.size() > 0);
  assert(input.size() == output.size());

  const float max = *std::max(input.begin(), input.end());

  float sum = 0.0f;
  for (int i = 0; i < input.size(); i++) {
    output[i] = std::exp(input[i] - max);
    sum += output[i];
  }

  for (int i = 0; i < input.size(); i++) {
    output[i] /= sum;
  }
}

template <class T>
void print(const std::string msg, const std::vector<T> data) {
  std::cout << msg << " : ";
  for (const auto e : data)
    std::cout << e << " ";
  std::cout << std::endl;
}

int main() {
  {
    std::vector<float> input = {1, 2, 3};
    std::vector<float> output(input.size());

    print("input", input);
    softmax(input, output);
    print("output", output);
  }
}
