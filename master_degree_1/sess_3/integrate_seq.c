#include <stdio.h>

int main() {
  unsigned long long steps = 100000000;
  double ans = 0.0;
  double top = 1.0;
  double segment = top / (double)steps;
  for (unsigned long long i = 0; i < steps; i++) {
    double x = segment * ((double)i + 0.5);
    double height = 4.0 / (1.0 + x*x);
    ans += height * segment;
  }
  printf("%lf", ans);
}
