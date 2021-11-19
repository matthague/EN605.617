__kernel void average_helper(__global * buffer) {
    size_t id = get_global_id(0);
		atomic_add(&buffer[0], buffer[id]);
		//buffer[0] = buffer[0] + buffer[id];
}
