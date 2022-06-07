import pycuda.autoinit
from pycuda import compiler, driver, gpuarray
import numpy as np


def do_recon_gpu(sino, w_sino, gamma_target_M, L2_M, gamma_coord, dbeta_proj):

    N_proj, cols = sino.shape
    N_matrix = L2_M.shape[1]
    gamma_max = np.max(gamma_coord)
    dgamma = gamma_coord[1]-gamma_coord[0]

    # block/thread allocation warning
    block_max=1024
    if N_matrix > block_max:
        print(f'need to manually set GPU block/thread for large matrix {N_matrix} > {block_max}')
    
    if np.log2(N_matrix)%1!=0:
        print(f'may need to manually set GPU block/thread for {N_matrix} size (not power of 2)')

    block_gpu=(N_matrix, block_max//N_matrix, 1)
    grid_gpu=(1,N_matrix//(block_max//N_matrix))
    #print('GPU block', block_gpu)
    #print('GPU grid', grid_gpu)
    
    kernel_code_template = """
        #include <math.h>
    
        __global__ void do_recon(float *matrix, float *sino, float *w_sino,  float *gamma_target_M, float *L2_M) {
            
            // get i, j for the matrix coordinate
            int i = threadIdx.x + blockDim.x * blockIdx.x;
            int j = threadIdx.y + blockDim.y * blockIdx.y;
       
            // assign constants
            int N_proj = %(N_PROJ)s;
            int N_matrix = %(N_MATRIX)s;
            int N_cols = %(COLS)s;
            float gamma_max = %(GAMMA_MAX)s; 
            float dgamma = %(DGAMMA)s;
            float dbeta_proj = %(DBETA_PROJ)s;

            // result at pixel (i,j)
            float result = 0.0;
            float w_result = 0.0; 
        
            for(int i_beta=0; i_beta < N_proj; i_beta++) {
                float L2 =           L2_M          [ i_beta*N_matrix*N_matrix + j*N_matrix + i ];
                float gamma_target = gamma_target_M[ i_beta*N_matrix*N_matrix + j*N_matrix + i ];
            
                if(fabsf(gamma_target) <  gamma_max) { 
                    int i_gamma0 = (int)((gamma_target + gamma_max)/dgamma);
                    float t = (dgamma*(i_gamma0+1) - gamma_max - gamma_target)/dgamma;
                
                    // linear interp
                    float this_q = (1-t)*sino[ i_beta*N_cols + i_gamma0] +   t*sino[ i_beta*N_cols  + i_gamma0 + 1];
                    float this_w = (1-t)*w_sino[ i_beta*N_cols + i_gamma0] + t*w_sino[ i_beta*N_cols + i_gamma0 + 1];
                
                    // add to results
                    result = result + (this_q * dbeta_proj / L2);
                    w_result = w_result + (this_w / L2);
                }
            }

            // write result to matrix
            matrix[ j * %(N_MATRIX)s + i ] = result/w_result;
        }
    """

    kernel_code = kernel_code_template % {
            'N_MATRIX':   N_matrix,
            'N_PROJ':     N_proj,
            'GAMMA_MAX':  gamma_max,
            'DGAMMA':     dgamma,
            'DBETA_PROJ': dbeta_proj,
            'COLS':       cols
            }

    # compile code
    mod = compiler.SourceModule(kernel_code)

    # get kernel function from compiled code
    do_recon_gpu = mod.get_function("do_recon")

    # move stuff to GPU
    sino = gpuarray.to_gpu(sino)
    w_sino = gpuarray.to_gpu(w_sino)
    gamma_target_M = gpuarray.to_gpu(gamma_target_M)
    L2_M = gpuarray.to_gpu(L2_M)

    # do the recon
    matrix = gpuarray.empty([N_matrix, N_matrix], np.float32)
    do_recon_gpu(matrix, sino, w_sino, gamma_target_M, L2_M, 
                 block=block_gpu, grid=grid_gpu)

    return matrix.get()

