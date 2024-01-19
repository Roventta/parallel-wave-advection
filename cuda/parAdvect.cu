// CUDA parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2021
// v1.0 15 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_, 
		   int verb) {
  M = M_, N = N_; Gx = Gx_; Gy = Gy_;  Bx = Bx_; By = By_; 
  verbosity = verb;
} //initParParams()


__host__ __device__
static void N2Coeff(double v, double *cm1, double *c0, double *cp1) {
  double v2 = v/2.0;
  *cm1 = v2*(v+1.0);
  *c0  = 1.0 - v*v;
  *cp1 = v2*(v-1.0);
}


struct info4T {
  int Tdim_x, Tdim_y, W_dim_x, W_dim_x_last, W_dim_y, W_dim_y_last;
  int Bdim_x, Bdim_y, BW_dim_x, BW_dim_x_last, BW_dim_y, BW_dim_y_last;
  int largestX, largestY;
};

struct info4T threadManager(int offset){
  struct info4T out;
  int N_ = N+offset;
  int M_ = M+offset;
  out.Tdim_x = Gx*Bx; out.Tdim_y = Gy*By;
  out.W_dim_x = (N_)/out.Tdim_x;
  out.W_dim_y = (M_)/out.Tdim_y;
  out.W_dim_x_last = out.W_dim_x+(N_)%out.Tdim_x;
  out.W_dim_y_last = out.W_dim_y+(M_)%out.Tdim_y;

  //case more thread than works
  if(out.W_dim_x==0){out.W_dim_x = 1; out.W_dim_x_last = 1;}
  if(out.W_dim_y==0){out.W_dim_y = 1; out.W_dim_y_last = 1;}

  out.Bdim_x = Gx; out.Bdim_y = Gy;
  out.BW_dim_x = (N_) / out.Bdim_x;
  out.BW_dim_y = (M_) / out.Bdim_y;
  
  out.BW_dim_x_last = out.BW_dim_x+(N_)%out.Bdim_x;
  out.BW_dim_y_last = out.BW_dim_y+(M_)%out.Bdim_y;
  return out;
}
__global__
void updateBoundaryEW2d(int M, int N, double*u, int ldu, struct info4T infoPack){
  int global_tid_x = threadIdx.x + blockIdx.x*blockDim.x;
  int global_tid_y = threadIdx.y + blockIdx.y*blockDim.y;

  int wsp_x = global_tid_x*infoPack.W_dim_x;
  int wsp_y = global_tid_y*infoPack.W_dim_y;

  //last 
  int workload_x = global_tid_x==infoPack.Tdim_x-1 ? infoPack.W_dim_x_last : infoPack.W_dim_x;
  int workload_y = global_tid_y==infoPack.Tdim_y-1 ? infoPack.W_dim_y_last : infoPack.W_dim_y;
 
  int i=0, j=0;

  for(int y=0; y<workload_y; y++){
    for(int x=0; x<workload_x; x++){
      i=wsp_y+y; j=wsp_x+x;
      if(j==0){V(u,i,0) = V(u,i,N);}
      if(j==N+1){V(u,i,N+1) = V(u,i,1);}
    }
  }

}
__global__
void updateBoundaryNS2d(int M, int N, double*u, int ldu, struct info4T infoPack){
  
  int global_tid_x = threadIdx.x + blockIdx.x*blockDim.x;
  int global_tid_y = threadIdx.y + blockIdx.y*blockDim.y;

  int wsp_x = global_tid_x*infoPack.W_dim_x;
  int wsp_y = global_tid_y*infoPack.W_dim_y;

  //last 
  int workload_x = global_tid_x==infoPack.Tdim_x-1 ? infoPack.W_dim_x_last : infoPack.W_dim_x;
  int workload_y = global_tid_y==infoPack.Tdim_y-1 ? infoPack.W_dim_y_last : infoPack.W_dim_y;
 
  int i=0, j=0;

  for(int y=0; y<workload_y; y++){
    for(int x=0; x<workload_x; x++){
      i=wsp_y+y; j=wsp_x+x;
      if(i==0){V(u,0,j) = V(u,M,j);}
      if(i==M+1){V(u,M+1,j) = V(u,1,j);}
    }
  }

}
__global__
void updateAdvectField2d_sharedMem(int M, int N, double* u, int ldu, double* v, int ldv, double Ux, double Uy, struct info4T infoPack) { 
  extern __shared__ double sharedMem[];

  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1);
  N2Coeff(Uy, &cjm1, &cj0, &cjp1);
  
  int my_block_size_x = blockIdx.x==infoPack.Bdim_x-1 ? infoPack.BW_dim_x_last:infoPack.BW_dim_x;
  int my_block_size_y = blockIdx.y==infoPack.Bdim_y-1 ? infoPack.BW_dim_y_last:infoPack.BW_dim_y;
  int my_block_sp_x = blockIdx.x*infoPack.BW_dim_x;
  int my_block_sp_y = blockIdx.y*infoPack.BW_dim_y;


  int workload_pT_x = my_block_size_x/blockDim.x;
  int workload_pT_y = my_block_size_y/blockDim.y;
  int blockLocal_sp_x = threadIdx.x*workload_pT_x;
  int blockLocal_sp_y = threadIdx.y*workload_pT_y;
  workload_pT_x += threadIdx.x == blockDim.x-1? my_block_size_x%blockDim.x:0;
  workload_pT_y += threadIdx.y == blockDim.y-1? my_block_size_y%blockDim.y:0;


  int global_wsp_x = blockLocal_sp_x + my_block_sp_x;
  int global_wsp_y = blockLocal_sp_y + my_block_sp_y;

  int sm_width = infoPack.BW_dim_x_last+2;
  
  int i=0, j=0;

  for(int y=0; y<workload_pT_y; y++){
    for(int x=0; x<workload_pT_x; x++){
      int global_i = global_wsp_y+y;
      int global_j = global_wsp_x+x;

      int sm_i = 1+blockLocal_sp_y+y;
      int sm_j = 1+blockLocal_sp_x+x;
      if(i < infoPack.largestY and j < infoPack.largestX){
      sharedMem[sm_i*sm_width+sm_j] = V(u,global_i, global_j);
      if(sm_i==1+my_block_size_y-1){sharedMem[(sm_i+1)*sm_width+sm_j] = V(u,global_i+1,global_j);}
      if(sm_j==1+my_block_size_x-1){sharedMem[sm_i*sm_width+sm_j+1] = V(u,global_i,global_j+1);}
      if(sm_i==1+0){sharedMem[(sm_i-1)*sm_width+sm_j] = V(u,global_i-1,global_j);}
      if(sm_j==1+0){sharedMem[sm_i*sm_width+sm_j-1] = V(u,global_i,global_j-1);}
      
      if(sm_i==1+0 and sm_j==1+0){sharedMem[(sm_i-1)*sm_width+sm_j-1] = V(u,global_i-1,global_j-1);}
      if(sm_i==1+0 and sm_j==1+my_block_size_x-1){sharedMem[(sm_i-1)*sm_width+sm_j+1] = V(u,global_i-1,global_j+1);}
      if(sm_i==1+my_block_size_y-1 and sm_j==1+0){sharedMem[(sm_i+1)*sm_width+sm_j-1] = V(u,global_i+1,global_j-1);}
      if(sm_i==1+my_block_size_y-1 and sm_j==1+my_block_size_x-1){sharedMem[(sm_i+1)*sm_width+sm_j+1] = V(u,global_i+1,global_j+1);}
      }
    }
  }

  __syncthreads();
  //if(blockIdx.x==0 and blockIdx.y ==0){printf("thread(%i %i) at last block, gonna be working (%i %i) pixels, starting from (%i %i)\n", threadIdx.x, threadIdx.y, workload_pT_x, workload_pT_y, global_wsp_x, global_wsp_y);}

  for (int y=0; y < workload_pT_y; y++){
    for (int x=0; x < workload_pT_x; x++){
      i = global_wsp_y+y;
      j = global_wsp_x+x;

      int sm_i = blockLocal_sp_y+y+1;
      int sm_j = blockLocal_sp_x+x+1;
      //if(blockIdx.x==2 and blockIdx.y==2){printf("mem: %f at (y:%i x:%i), true: %f at (y:%i x:%i)\n", sharedMem[(sm_i+1)*sm_width+sm_j+1],sm_i+1, sm_j+1, V(u,i+1,j+1), i+1, j+1);}
     
      if(i<infoPack.largestY and j<infoPack.largestX){
      V(v,i,j) =
      
      cim1*(cjm1*sharedMem[(sm_i-1)*sm_width+sm_j-1] + cj0*sharedMem[(sm_i-1)*sm_width+sm_j] + cjp1*sharedMem[(sm_i-1)*sm_width+sm_j+1]) +
      ci0 *(cjm1*sharedMem[(sm_i)*sm_width+sm_j-1]+ cj0*sharedMem[sm_i*sm_width+sm_j] + cjp1*sharedMem[(sm_i)*sm_width+sm_j+1]) +
      cip1*(cjm1*sharedMem[(sm_i+1)*sm_width+sm_j-1] + cj0*sharedMem[(sm_i+1)*sm_width+sm_j] + cjp1*sharedMem[(sm_i+1)*sm_width+sm_j+1]);
      /*
      cim1*(cjm1*V(u,i-1,j-1) + cj0*sharedMem[(sm_i-1)*sm_width+sm_j] + cjp1*V(u,i-1,j+1)) +
      ci0 *(cjm1*sharedMem[(sm_i)*sm_width+sm_j-1]+ cj0*sharedMem[sm_i*sm_width+sm_j] + cjp1*sharedMem[(sm_i)*sm_width+sm_j+1]) +
      cip1*(cjm1*V(u,i+1,j-1) + cj0*sharedMem[(sm_i+1)*sm_width+sm_j] + cjp1*V(u,i+1,j+1));
      */
      }
    }
  }
}


__global__
void updateAdvectField2d(int M, int N, double* u, int ldu, double* v, int ldv, double Ux, double Uy, struct info4T infoPack) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1);
  N2Coeff(Uy, &cjm1, &cj0, &cjp1);

  int global_tid_x = threadIdx.x + blockIdx.x*blockDim.x;
  int global_tid_y = threadIdx.y + blockIdx.y*blockDim.y;

  int wsp_x = global_tid_x*infoPack.W_dim_x;
  int wsp_y = global_tid_y*infoPack.W_dim_y;
 
  //last 
  int workload_x = global_tid_x==infoPack.Tdim_x-1 ? infoPack.W_dim_x_last : infoPack.W_dim_x;
  int workload_y = global_tid_y==infoPack.Tdim_y-1 ? infoPack.W_dim_y_last : infoPack.W_dim_y;

  //if(blockIdx.x==0 and blockIdx.y==0){
  //printf("from (x=%i,y=%i), need to work (%i, %i), %i, %i\n", wsp_x, wsp_y, workload_x, workload_y, global_tid_x, infoPack.Tdim_x);
  //}
  int i = 0, j = 0;

  for (int y=0; y < workload_y; y++){
    for (int x=0; x < workload_x; x++){
      i = wsp_y+y;
      j = wsp_x+x;
      if(i < infoPack.largestY and j < infoPack.largestX){
      V(v,i,j) =
      cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
      ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
      cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
      }
    }
  }
}
__global__ 
void copyField2d(int M, int N, double *u, int ldu, double *v, int ldv, struct info4T infoPack) {
  int global_tid_x = threadIdx.x + blockIdx.x*blockDim.x;
  int global_tid_y = threadIdx.y + blockIdx.y*blockDim.y;

  int wsp_x = global_tid_x*infoPack.W_dim_x;
  int wsp_y = global_tid_y*infoPack.W_dim_y;
 
  //last 
  int workload_x = global_tid_x==infoPack.Tdim_x-1 ? infoPack.W_dim_x_last : infoPack.W_dim_x;
  int workload_y = global_tid_y==infoPack.Tdim_y-1 ? infoPack.W_dim_y_last : infoPack.W_dim_y;

  int x=0, y=0;
  //printf("from (x=%i,y=%i), need to work (%i, %i), %i, %i\n", wsp_x, wsp_y, workload_x, workload_y, global_tid_x, infoPack.Tdim_x);
  for(int i=0; i<workload_y; i++){
    for(int j=0; j<workload_x; j++){
      x = wsp_x+j; y=wsp_y+i;
      v[y*ldu + x] = u[y*ldv + x];}
  }

}

// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int reps, double *u, int ldu) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N+2; double *v;
  struct info4T infoPack = threadManager(0);
  struct info4T infoPackBd = threadManager(2*1);

  infoPack.largestX = N+2;
  infoPack.largestY = M+2;
  int largestBlock = (infoPack.BW_dim_x_last+2) * (infoPack.BW_dim_y_last+2);
  printf("sharedMem size is %i double\n", largestBlock);
  if(largestBlock >= 6144){printf("shreadMem block size is greater than hard ware limitiation, will have garbage outputs\n");}
  dim3 block(Bx, By);
  dim3 grid(Gx, Gy);
  HANDLE_ERROR( cudaMalloc(&v, ldv*(M+2)*sizeof(double)) );
  for (int r = 0; r < reps; r++) {
    //updateBoundaryNS <<<1,1>>> (N, M, u, ldu);
    updateBoundaryNS2d <<<grid,block>>> (M,N,u,ldu, infoPackBd);
    updateBoundaryEW2d <<<grid,block>>> (M, N, u, ldu, infoPackBd);
    //updateBoundaryEW <<<1,1>>> (M, N, u, ldu);
    //updateAdvectFieldK <<<1,1>>> (M, N, &V(u,1,1), ldu, &V(v,1,1), ldv, Ux, Uy);
    //updateAdvectField2d <<<grid,block>>> (M, N, &V(u,1,1), ldu, &V(v,1,1), ldv, Ux, Uy, infoPack);
    updateAdvectField2d_sharedMem<<<grid,block, largestBlock*sizeof(double)>>>(M, N, &V(u,1,1), ldu, &V(v,1,1), ldv, Ux, Uy, infoPack);
    //copyFieldK <<<1,1>>> (M, N, &V(v,1,1), ldv, &V(u,1,1), ldu);
    copyField2d <<<grid,block>>> (M, N, &V(v,1,1), ldv, &V(u,1,1), ldu, infoPack);
  } //for(r...)
  HANDLE_ERROR( cudaFree(v) );
} //cuda2DAdvect()



// ... optimized parallel variant
void cudaOptAdvect(int reps, double *u, int ldu, int w) {

} //cudaOptAdvect()
