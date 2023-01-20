// atomicMax is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMaxf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));        
    } else {
        return __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
    }
}

__device__ unsigned int get_strided_index(
    unsigned int idx,
    size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

__device__ unsigned int get_unstrided_index(
    const unsigned int strided_i,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int idx = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        idx *= dims[d];
        idx += strides[d] == 0 ? 0 : (strided_i / strides[d]) % dims[d];
    }
    return idx;
}

extern "C" __global__ void fill_with(float *buf, float value, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    buf[i] = value;
}

// Sourced from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
__device__ __forceinline__ unsigned int next_power_of_two(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v++;
    return v;
}

// Efficiently computes the max of each chunk in "data" of size chunk_len, and
// stores the maximums in out[i / chunk_len]
__device__ void chunk_max(
    const size_t numel,
    const size_t chunk_len,
    const float data,
    float* out
) {
    __shared__ float buf[1024];
    // assumes that threads where i >= numel have already exited
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int block_i = threadIdx.x;
    buf[block_i] = data;

    unsigned int chunk_i = i % chunk_len;
    unsigned int chunk_start = max((int)(block_i - chunk_i), 0);
    unsigned int chunk_end = min((unsigned int)(block_i + chunk_len - chunk_i), blockDim.x);

    chunk_i = block_i - chunk_start;

    size_t max_chunk_len = min(chunk_end - chunk_start, blockDim.x);
    size_t incr = next_power_of_two(max_chunk_len) >> 1;

    __syncthreads();

    // Uses sequential addressing as discussed in
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    for (; incr > 0; incr >>= 1) {
        unsigned int block_i_2 = block_i + incr;

        if (block_i_2 < chunk_end && chunk_i < incr) {
            // This is sound because __syncthreads and the conditions above
            // ensure that no data races occur
            buf[block_i] = fmaxf(buf[block_i], buf[block_i_2]);
        }

        __syncthreads();
    }

    if (block_i == chunk_start) {
        atomicMaxf(out + i / chunk_len, buf[block_i]);
    }
}

// strides and dims specify how to index inp to put all summed elements next to
// each other, and chunk_len is len(inp) / len(out)
extern "C" __global__ void max_to_forward(
    const size_t numel,
    const size_t num_dims,
    const size_t chunk_len,
    const float *inp,
    const size_t *dims,
    const size_t *strides,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    unsigned int inp_i = get_strided_index(i, num_dims, dims, strides);
    chunk_max(numel, chunk_len, inp[inp_i], out);
}

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
extern "C" __global__ void max_to_backward(
    const size_t numel,
    const size_t num_dims,
    const float elems_per_thread,
    const size_t *dims,
    const float *inp,
    float *grad_inp,
    const size_t *inp_strides,
    const float *out,
    const float *grad_out,
    const size_t *out_strides
) {
    unsigned int inp_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (inp_i >= numel) {
        return;
    }

    unsigned int i = get_unstrided_index(inp_i, num_dims, dims, inp_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);

    auto tmp = inp[inp_i] == out[out_i] ? grad_out[out_i] : 0.0;
    grad_inp[inp_i] += tmp * elems_per_thread;
}
