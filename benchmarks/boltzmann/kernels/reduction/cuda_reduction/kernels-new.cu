#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define NN 32

namespace py = pybind11;

__global__ void redcheck(double *w, int r,int RM,int M)
{
int i=threadIdx.x+blockIdx.x*blockDim.x;
int j;
int tid=threadIdx.x;
if(i<M)
{
    //create shared memory(dynamic) and load elements
    extern __shared__ double bb[];
    for (j=0;j<r;j++)
    {
        bb[r*tid+j]=w[r*i+j];
    }

    __syncthreads();
   
   //do reduction per block.
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
         {
            if (tid < s) 
            {
                 for(j=0;j<r;j++)
                 {
                     bb[r*tid+j] += bb[r*(tid + s)+j];
                 }
            
            }
              __syncthreads();
        }


//each block writes results.
if (tid == 0)
{
     for(j=0;j<r;j++)
     {
         w[r*blockIdx.x+j]=bb[j];
     }
}


}
}





__global__ void reda(double *w, double *v, double *x, int r,int p,int M)
{
int i=threadIdx.x+blockIdx.x*blockDim.x;
int j=threadIdx.y+blockIdx.y*blockDim.y;
//printf("gpu i %d\n",i);
if(i<p&&j<r)
{
int b=int(x[i]*float(M));

atomicAdd(&w[r*b+j],v[r*i+j]);
/*
if((r*b+j)>(r*M))
{
    printf("this is bad %d",r*b+j);
}
*/
}
}


void calls(
    int r,int p,int M,int RM,
    py::array_t<double>& dw_arr, py::array_t<double>& dv_arr, py::array_t<double>& dx_arr, int ptcl, uintptr_t stream_ptr)
{

    void * temp = reinterpret_cast<void*>(stream_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(temp);

    int blockx=256;
    int blocky=r;
  
    dim3 tpb(blockx,blocky,1);
    dim3 bpg(ceil(ptcl/blockx)+1,ceil(r/blocky)+1,1);

    int bloks = ceil((double)M/RM );
    size_t ns = bloks*r*sizeof(double);

    dim3 tpb2(bloks,1,1);
    dim3 bpg2(ceil(M/bloks)+1,1,1);


    double* dw = static_cast<double *>(dw_arr.request().ptr);
    double* dv = static_cast<double *>(dv_arr.request().ptr);
    double* dx = static_cast<double *>(dx_arr.request().ptr);


    reda<<<bpg,tpb,0,stream>>>(dw,dv,dx,r,ptcl,M);

   // cudaDeviceSynchronize();
    if ((M/RM) > 1){
    redcheck<<<bpg2,tpb2,ns,stream>>>(dw,r,RM,M);
    }
   // cudaDeviceSynchronize();

}

PYBIND11_MODULE(cuda_red, m) {
    m.def("calls", &calls);
}
