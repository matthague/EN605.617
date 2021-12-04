__kernel void bubble_sort(__global int *a, int max_size, int parity) {
    int i = get_global_id(0);
    if ((i + 1 < max_size) && (i % 2 == parity)) {
        if (a[i] > a[i + 1]) {
            int temp = a[i];
            a[i] = a[i + 1];
            a[i + 1] = temp;
        }
    }
}
