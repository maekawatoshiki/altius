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

// https://arxiv.org/pdf/2101.01321.pdf
std::tuple<std::vector<int8_t>, float>
softmax2(const std::vector<int8_t> &input, const float scale) {
  assert(input.size() > 0);

  constexpr int n = 2;
  const int8_t max = *std::max_element(input.begin(), input.end());
  std::vector<int8_t> exp(input.size());
  std::vector<int8_t> output(input.size());
  const float scale_exp = scale / (1 << n);

#define P(x) std::cout << #x << " = " << (int)x << std::endl;

  const float a = 0.3585, b = 1.353, c = 0.344;
  auto poly = [&](int32_t q, float scale) -> std::tuple<int32_t, float> {
    const int32_t q_b = b / scale;
    const int32_t q_c = c / (scale * scale);
    const float scale_out = a * scale * scale;
    const int32_t q_out = (q + q_b) * q + q_c;
    return std::make_tuple(q_out, scale_out);
  };

  float scale_out;
  for (int i = 0; i < input.size(); i++) {
    const int32_t x = input[i] - max;
    const int32_t x_ln2 = std::log(2.f) / scale;
    const int32_t z = -x / x_ln2;
    const int32_t x_p = x + z * x_ln2;
    const auto [q_l, scale_l] = poly(x_p, scale);
    scale_out = scale_l;
    const int32_t x_exp = q_l >> z;
    P(z);
    P(x_p);
    exp[i] = x_exp;
    P(exp[i]);
  }
  P(scale_out);

  int32_t sum = 0;
  for (int i = 0; i < input.size(); i++) {
    sum += exp[i];
  }
  P(sum);

  constexpr int m = 8;

  for (int i = 0; i < input.size(); i++) {
    output[i] = exp[i] / sum;
    P(output[i]);
  }

  return std::make_tuple(output, scale_out);
}

std::tuple<std::vector<int8_t>, float> softmax(const std::vector<int8_t> &input,
                                               const float scale) {
  assert(input.size() > 0);

  // constexpr int n = 2;
  const int8_t max = *std::max_element(input.begin(), input.end());
  std::vector<int8_t> exp(input.size());
  std::vector<int8_t> output(input.size());
  // const float scale_exp = scale / (1 << n);
  const int8_t x_0 = (int8_t)std::round(1.f / scale);

#define P(x) std::cout << #x << " = " << (int)x << std::endl;

  for (int i = 0; i < input.size(); i++) {
    const int8_t x = input[i] - max;
    // P(x);
    const int8_t x_p = x + (x >> 1) - (x >> 4);
    // P(x_p);
    // P(x_0);
    const int8_t q = -x_p / x_0;
    // P(q);
    const int8_t r = -(x_p - q * (-x_0));
    // P(r);
    const int8_t x_b = ((-r) >> 1) + x_0;
    // P(x_b);
#if 0
    int N = 3;
    if (N - q > 0) {
      exp[i] = x_b << (N - q); // TODO
    } else {
      exp[i] = x_b >> (q - N); // TODO
    }
#else
    exp[i] = x_b >> q;
#endif
    // P(x_b);
    // P(q);
    // P(exp[i]);
    // std::cout << (int)exp[i] << std::endl;
  }

  int32_t sum = 0;
  for (int i = 0; i < input.size(); i++) {
    sum += exp[i];
  }
  // P(sum);

  constexpr int m = 28;
  constexpr int bits = 8;

  for (int i = 0; i < input.size(); i++) {
    output[i] = (((1 << m) / sum) * exp[i]) >> (m - (bits - 1));
    P(output[i]);
  }

  return std::make_tuple(output, 1.f / (float)(1 << (bits - 1)));
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
  const std::vector<float> input = {-0.5, 0, 0.1, 0.15, 0.2, 0.3, 0.35, 0.9, 1.01};
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
  // {
  //   std::cout << "# QOp" << std::endl;
  //   const auto output = softmax2(quantize(input, scale), scale);
  //   print("input", input);
  //   print("output", dequantize(std::get<0>(output), std::get<1>(output)));
  // }
}
