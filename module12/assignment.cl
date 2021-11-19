__kernel void average(__global * buffer) {
    size_t id = get_global_id(0);
    buffer[id] = (buffer[id] + buffer[id + 1] + buffer[id + 2] + buffer[id + 3]) / 4;
}
