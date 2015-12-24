
from __future__ import print_function
import pyopencl as cl
import numpy as np
import time

size = 256
ha = size
wa = size
wb = size

ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(ctx)
a = np.random.random((ha, wa)).astype(np.float32)
b = np.random.random((wa, wb)).astype(np.float32)
c = np.zeros((ha, wb)).astype(np.float32)

a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, c.nbytes)

code = '''
#define BLOCK_SIZE 16

//////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
//! gpgpu-computing4.blogspot.co.uk/2009/10/matrix-multiplication-3-opencl.html
//////////////////////////////////////////////////////
__kernel void
matrixMul(__global float* C,
          __global float* A,
          __global float* B, const int wA, const int wB)
{
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed
    // by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed
    // by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the
    // sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed
    // by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the
    // sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep)
    {

        // Declaration of the local memory array As
        // used to store the sub-matrix of A
        __local float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the local memory array Bs
        // used to store the sub-matrix of B
        __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;

}
'''

prg = cl.Program(ctx, code).build()
gsize = (size, size)
lsize = (16, 16)

for i in range(2000):
    start = time.time()
    event = prg.matrixMul(queue, gsize, lsize, c_buf, a_buf, b_buf, np.int32(wa), np.int32(wb))
    event.wait()
    gputime = time.time() - start
    cl.enqueue_copy(queue, c, c_buf)

    start = time.time()
    npresult = np.dot(a, b)
    cputime = time.time() - start

    print('err={}, g={}, c={}'.format(np.count_nonzero(1 - (c == npresult)), gputime, cputime))

print('done')
