#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void backward_kernel(const float* Q, const float* K, const float* V, const float* Mask ,const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* ){
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset = (bx * gridDim.y * N) + (by * N); 

    extern__shared__float sram[];

    int tile_size =Bc * d;
    
    float* Qi =sram;
    float* Kj =&sram[tile_size];
    float* Vj =&sram[tile_size * 2];
    float* S = &sram[tile_size * 3];
    float* dKj =&sram[tile_size * 4];
    float* dVj =&sram[tile_size * 5];
    float* dOi =&sram[tile_size * 6];
    float* dQi = &sram[tile_size * 7];
    float* dOi = &sram[tile_size * 8];
    float* Pi  = &sram[]; 
    float* Di  = &sram[]; 
    for(int j=0;j<Tr;j++){
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            dKj[(tx * d) + x] = 0;
            dVj[(tx * d) + x] = 0;
        }

        __syncthreads(); 

        for(int i=0;i<Tc;i++){
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
                dQi[(tx * d) + x] = dQ[qkv_offset + (tile_size * i) + (tx * d) + x];
                Oi[(tx * d) + x] = O[qkv_offset + (tile_size * i) + (tx * d) + x];
            }

            //compute Si,Pi
            for (int y = 0; y < Br; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Br * tx) + y] = sum;
                P[(Br * tx) + y] = __expf(S[(Br * tx) + y]-Li[tx]);
            }
            //compute dVj Br*Bc*Bc*d
            //default Bc = Br  or pay attention on tx and Br

            for(int y = 0; y <d ;y++){
                float sum = 0;
                for(int x = 0; x < Bc;x++){
                    sum +=Pi[(tx * Bc) + x] * dOi[( x * d) + y];
                }
                dVj[(tx * d) + y] =  dVj[(tx * d) + y] + sum;
            }

            //compute dPi

            for(int y =0; y<Br;y++){
                float sum =0;
                for(int x =0; x<d;x++){
                    sum +=dOi[(tx + d) +x] * Vj[(y * d) + x];
                }
                dPi[(Br * tx) + y] = sum;
            }

            //compute dSi
            for(int y =0 ; y<Br ; y++){
                dSi[(tx * Br) + y] = P[(tx * Br) + y] * ( dPi[(Br * tx) + y] - Di[tx]);
            }

            //compute dQi Bc*Br*Br*d
            for(int y = 0; y <d ;y++){
                float sum = 0;
                for(int x = 0; x < Br;x++){
                    sum +=dSi[(tx * Br) + x] * Kj[(x * d) + y];
                }
                dQ[qkv_offset + (tile_size * i) + (tx * d) + y] =  dQi[(tx * d) + y] + sum;
            }

            //compute dKj Br*Bc*Bc*d
            for(int y = 0; y <d ;y++){
                float sum = 0;
                for(int x = 0; x < Bc;x++){
                    sum +=dSi[(tx * Bc) + x] * Qj[(x * d) + y];
                }
                dKj[(tx * d) + y] =  dKj[(tx * d) + y] + sum;
            }
        }

    for (int x = 0; x < d; x++) {
            dK[qkv_offset + (tile_size * j) + (tx * d) + x] = Kj[(tx * d) + x];
            dV[qkv_offset + (tile_size * j) + (tx * d) + x] = vj[(tx * d) + x];
        }

    }

}


torch::Tensor backward(torch::Tensor Q,torch::Tensor K,torch::Tensor V,torch:::Tensor dO, torch:::Tensor L,torch:::Tensor D){
    const int Bc = 32; 
    const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    auto dQ =torch::zeros_like(Q);
    auto dK =torch::zeros_like(K);
    auto dV =torch::zeros_like(V);
    
    //alloc block memory
    const int sram_size = ;


    dim3 grid_dim(B,nh);
    dim3 block_dim(Bc);


    backward_kernel<<<grid_dim,block_dim,sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),Mask.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    )

    std::pair<torch::Tensor, torch::Tensor,torch::Tensor> result(dQ,dK,dv);
    return result;
}