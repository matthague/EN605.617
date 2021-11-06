__kernel void add_kernel(__global const float *a, __global const float *b, __global float *result) {
    int idx = get_global_id(0);
    result[idx] = a[idx] + b[idx];
}

__kernel void sub_kernel(__global const float *a, __global const float *b, __global float *result) {
    int idx = get_global_id(0);
    result[idx] = a[idx] - b[idx];
}

__kernel void mult_kernel(__global const float *a, __global const float *b, __global float *result) {
    int idx = get_global_id(0);
    result[idx] = a[idx] * b[idx];
}

__kernel void div_kernel(__global const float *a, __global const float *b, __global float *result) {
    int idx = get_global_id(0);
    result[idx] = a[idx] / b[idx];
}

__kernel void pow_kernel(__global const float *a, __global const float *b, __global float *result) {
    int idx = get_global_id(0);
    result[idx] = pow(a[idx], b[idx]);
}
