#include <stdio.h>
#include <assert.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
  unsigned int frac : 23;
  unsigned int exp  : 8;
  unsigned int sign : 1;
} f32;

float add(float x, float y) {
  const f32 xi = *(f32*)&x;
  const f32 yi = *(f32*)&y;
  printf("xi.exp = %d, yi.exp = %d\n", xi.exp, yi.exp);
  unsigned int exp = MAX(xi.exp, yi.exp);
  unsigned int xi_frac = (xi.frac | (1 << 23)) >> (exp - xi.exp);
  unsigned int yi_frac = (yi.frac | (1 << 23)) >> (exp - yi.exp);
  const int carry = (xi_frac + yi_frac) > 0xffffff;

  const f32 zi = {
    .sign = 0,
    .exp  = exp + carry,
    .frac = ((xi_frac + yi_frac) >> carry) & 0x7fffff,
  };
  return *(float*)&zi;

#if 0
  const int xi = *(int*)&x;
  const int yi = *(int*)&y;
  const int xi_sign = (xi >> 31) & 0x1;
  const int xi_exp = (xi >> 23) & 0x7f;
  const int xi_frac = xi & 0x7fffff;
  const int yi_sign = (yi >> 31) & 0x1;
  const int yi_exp = (yi >> 23) & 0x7f;
  const int yi_frac = yi & 0x7fffff;
  assert(xi_sign == 0);
  assert(yi_sign == 0);
  assert(xi_exp == 127);
  assert(yi_exp == 127 - 1);
  /* assert(xi_frac == 0); */
  /* assert(yi_frac == 0); */
  const int zi_sign = 0;
  const int zi_exp = 127;
  const int zi_frac = xi_frac + (yi_frac >> 1) + 0x400000;
  const int zi = (zi_sign << 31) | (zi_exp << 23) | zi_frac;
  return *(float*)&zi;
#endif
}

int main() {
  /* 1 + 8 + 7 */
  /* sign exp frac */
  float x = 10.2;
  float y = 0.1;
  float z = add(x, y);
  printf("%f\n", z);
  return 0;
}
