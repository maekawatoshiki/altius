#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

int8_t quantize(float x, float scale) { return x / scale; }

float dequantize(int8_t x, float scale) { return (float)x * scale; }

std::vector<int8_t> quantize(const std::vector<float> &x, float scale) {
  std::vector<int8_t> output(x.size());
  for (int i = 0; i < x.size(); i++) {
    output[i] = x[i] / scale;
  }
  return output;
}

std::vector<float> dequantize(const std::vector<int8_t> &x, float scale) {
  std::vector<float> output(x.size());
  for (int i = 0; i < x.size(); i++) {
    output[i] = (float)x[i] * scale;
  }
  return output;
}

std::tuple<std::vector<int8_t>, float> softmax(const std::vector<int8_t> &input,
                                               const float scale) {
  assert(input.size() > 0);

  constexpr int n = 2;
  const int8_t max = *std::max_element(input.begin(), input.end());
  std::vector<int8_t> exp(input.size());
  std::vector<int8_t> output(input.size());
  const float scale_exp = scale / (1 << n);

#define P(x) std::cout << #x << " = " << (int)x << std::endl;

  for (int i = 0; i < input.size(); i++) {
    const auto x = input[i] - max;
    // P(x);
    const auto x_p = x + (x >> 1) - (x >> 4);
    // P(x_p);
    const auto x_0 = (int32_t)std::round(1.f / scale);
    // P(x_0);
    const auto q = x_p / (-x_0);
    // P(q);
    const auto r = -(x_p - q * (-x_0));
    // P(r);
    const auto x_b = ((-r) >> 1) + x_0;
    // P(x_b);
#if 0
    int N = 0;
    if (N - q > 0) {
      exp[i] = x_b << (N - q); // TODO
    } else {
      exp[i] = x_b >> (q - N); // TODO
    }
#else
    exp[i] = x_b >> q;
#endif
    P(x_b);
    P(q);
    P(exp[i]);
    // std::cout << (int)exp[i] << std::endl;
  }

  int32_t sum = 0;
  for (int i = 0; i < input.size(); i++) {
    sum += exp[i];
  }
  // P(sum);

  constexpr int m = 8;

  for (int i = 0; i < input.size(); i++) {
    output[i] = (((1 << m) / sum) * exp[i]) >> (m - (8 - 1));
    // P(output[i]);
  }

  return std::make_tuple(output, 1.f / (float)(1 << (8 - 1)));
}

std::vector<float> softmax(const std::vector<float> &input) {
  assert(input.size() > 0);

  std::vector<float> output(input.size());
  const float max = *std::max_element(input.begin(), input.end());

  float sum = 0.0f;
  for (int i = 0; i < input.size(); i++) {
    output[i] = std::exp(input[i] - max);
    sum += output[i];
  }

  for (int i = 0; i < input.size(); i++) {
    output[i] /= sum;
  }

  return output;
}

void print(const std::string msg, const std::vector<int8_t> data) {
  std::cout << msg << " : ";
  for (const auto e : data)
    std::cout << (int)e << " ";
  std::cout << std::endl;
}

void print(const std::string msg, const std::vector<float> data) {
  std::cout << msg << " : ";
  for (const auto e : data)
    std::cout << e << " ";
  std::cout << std::endl;
}

int main() {
  const std::vector<float> input = {-1, 0, 1, 2, 3, 4, 5};
  const float scale = 0.04;
  {
    print("input", input);
    auto output = softmax(input);
    print("output", output);
  }
  {
    print("input", input);
    print("input(int)", quantize(input, scale));
    print("output", dequantize(quantize(input, scale), scale));
  }
  {
    std::cout << "# QDQ" << std::endl;
    const auto output =
        dequantize(quantize(softmax(dequantize(quantize(input, scale), scale)),
                            1.f / (1 << 7)),
                   1.f / (1 << 7));
    print("input", input);
    print("output", output);
  }
  {
    std::cout << "# QOp" << std::endl;
    const auto output = softmax(quantize(input, scale), scale);
    print("input", input);
    print("output", dequantize(std::get<0>(output), std::get<1>(output)));
  }
}
