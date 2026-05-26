#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include "gelu_omp.h"

inline float gelu_fast_rational(float x) {
    float abs_x = std::abs(x);

    const float c1 = 0.200855f;
    const float c2 = 0.109956f;
    const float c3 = 0.000390f;
    const float c4 = 0.020427f;

    float x2 = x * x;
    float abs_x3 = abs_x * x2;
    float x4 = x2 * x2;
    
    float poly = 1.0f + c1 * abs_x + c2 * x2 + c3 * abs_x3 + c4 * x4;

    float poly2 = poly * poly;
    float poly4 = poly2 * poly2;
    float inv_poly4 = 1.0f / poly4;

    float cdf_poly = 0.5f * inv_poly4;

    float sign = std::copysign(1.0f, x);
    float cdf = 0.5f + sign * (0.5f - cdf_poly);
    
    return x * cdf;
}

inline float fast_tanh(float x) {
    float e2x = std::exp(2.f * x);
    return (e2x - 1.f) / (e2x + 1.f);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float* inptr = input.data();
    float* outptr = output.data();

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        float x = inptr[i];
        // float inner = 0.79788456f * x * (1.f + 0.044715f * x * x);
        // outptr[i] = 0.5f * x * (1.f + fast_tanh(inner));
        outptr[i] = gelu_fast_rational(x);
    }

    return output;
}

#if 1
std::vector<float> GeluRef(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    constexpr float argscale = std::sqrt(2.f/M_PI);
    const float* inptr = input.data();
    float* outptr = output.data();

    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        float y = 0.5f*x*(1 + std::tanh(argscale*x*(1.f + 0.044715f*x*x)));
        output[i] = y;
    }

    return output;
}

int main() {
    size_t n = 134217728u;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    auto y = GeluOMP(x);

    std::vector<float> yref = GeluRef(x);
    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(y[i] - yref[i]));
    }
    printf("max absolute error = %.5g\n", err);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto y = GeluOMP(x);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.2f\n", time);

    return 0;
}
#endif
