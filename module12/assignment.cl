__kernel void average(__global * buffer) {
    size_t id = get_global_id(0);
		float avg;
		if(id == 15) {
				avg = ((float)(buffer[id] + buffer[id - 1])) / 2;
		} else if (id == 0) {
				avg = ((float)(buffer[id] + buffer[id + 1])) / 2;
		} else {
		    avg = ((float)(buffer[id] + buffer[id - 1] + buffer[id + 1])) / 3;
		}
		// rounding without math.h
		int sign = (int)((avg > 0) - (avg < 0));
    int odd = ((int)avg % 2); // odd -> 1, even -> 0
    buffer[id] = (int)(avg-sign*(0.5f-odd));
}
