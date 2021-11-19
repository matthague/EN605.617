__kernel void average(__global * buffer) {
    size_t id = get_global_id(0);
		if(id == 15) {
				buffer[id] = (buffer[id] + buffer[id - 1] + buffer[id - 1]) / 3;
		} else if (id == 0) {
				buffer[id] = (buffer[id] + buffer[id + 1] + buffer[id + 1]) / 3;
		} else {
		    buffer[id] = (buffer[id] + buffer[id - 1] + buffer[id + 1]) / 3;
		}
}
