#include <quadmath.h>
#include <stdio.h>

#include <vector>
typedef __float128 quad;
int main() {
    __float128 a = 1.0Q, b = 3.14159265358979323846Q;
    __float128 c = a + b;

    std::vector<quad> vec1 = {a, b, c};
    std::vector<quad> vec2 = {a, b, c};

    std::vector<quad> sum_vec(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        sum_vec[i] = vec1[i] + vec2[i];
    }

    // Print the result
    printf("Sum of vectors:\n");
    for (const auto& val : sum_vec) {
        char buf[128];
        quadmath_snprintf(buf, sizeof(buf), "%.36Qg", val);
        printf("%s ", buf);
    }
    char buf[128];
    quadmath_snprintf(buf, sizeof(buf), "%.36Qg", c);
    printf("Result: %s\n", buf);

    return 0;
}
