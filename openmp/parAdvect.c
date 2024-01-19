// OpenMP parallel 2D advection solver module
// template written for COMP4300/8300 Assignment 2, 2021
// template v1.0 14 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "serAdvect.h" // advection parameters

static int M, N, P, Q; // local store of problem parameters
static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb) {
  M = M_, N = N_; P = P_, Q = Q_;
  verbosity = verb;
} //initParParams()


static void omp1dUpdateBoundary(double *u, int ldu) {
  int i, j;
  for (j = 1; j < N+1; j++) { //top and bottom halo
    V(u, 0, j)   = V(u, M, j);
    V(u, M+1, j) = V(u, 1, j);
  }
  for (i = 0; i < M+2; i++) { //left and right sides of halo 
    V(u, i, 0) = V(u, i, N);
    V(u, i, N+1) = V(u, i, 1);
  }
} 


static void omp1dUpdateAdvectField(double *u, int ldu, double *v, int ldv) {
  int i, j;
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1); N2Coeff(Uy, &cjm1, &cj0, &cjp1);
  int n = omp_get_num_procs();
  int blockSize = M/n;
	#pragma omp parallel for default(shared) schedule(static, blockSize) 
	for (i=0; i < M; i++){
		for (j=0; j < N; j++){
			V(v,i,j) =
			cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
			ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
			cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
		}	
	}
} //omp1dUpdateAdvectField()  


static void omp1dCopyField(double *v, int ldv, double *u, int ldu) {
  int i, j;
  int n = omp_get_num_procs();
  int blockSize = M/n;
  #pragma omp parallel for default(shared) schedule(static, blockSize) 
  for (i=0; i < M; i++)
    for (j=0; j < N; j++)
      V(u,i,j) = V(v,i,j);
} //omp1dCopyField()


// evolve advection over reps timesteps, with (u,ldu) containing the field
// using 1D parallelization
void omp1dAdvect(int reps, double *u, int ldu) {
  int r, ldv = N+2;
  double *v = calloc(ldv*(M+2), sizeof(double)); assert(v != NULL);
  for (r = 0; r < reps; r++) {    
    omp1dUpdateBoundary(u, ldu);
    omp1dUpdateAdvectField(&V(u,1,1), ldu, &V(v,1,1), ldv);
    omp1dCopyField(&V(v,1,1), ldv, &V(u,1,1), ldu);
  } //for (r...)
  free(v);
} //omp1dAdvect()

void omp2dBorderTB(double*u, int ldu, int bgp, int mloc, int nloc, int p, int q,int P, int Q){
	int  j;
	int localDim_n = (N+2)/Q;
	int blockIndex_n = q*localDim_n;

	//horizontal borders
	int workStart_h = 0;
	if(q == 0){nloc-=1;workStart_h +=1;}
	if(q == Q-1){nloc-=1;}
	if(p == 0){
		for(j=0; j<nloc; j++){
		//printf("%i %i %i\n",workStart_h, nloc, blockIndex_n+j);
		V(u, 0, blockIndex_n+j+workStart_h) = V(u, M, blockIndex_n+j+workStart_h);
		}
	}
	
	if(p == P-1){
		for(j=0;j<nloc;j++){
		V(u, M+1, blockIndex_n+j+workStart_h) = V(u, 1, blockIndex_n+j+workStart_h); 
		}
	}

}
void omp2dBorderLR(double*u, int ldu, int bgp, int mloc, int nloc, int p, int q,int P, int Q){
	int i;
	int localDim_n = (N+2)/Q;
	int blockIndex_n = q*localDim_n;

	int localDim_m = (M+2)/P;
	int blockIndex_m = p*localDim_m;
	//vertical borders, no start or end manipulations
	if(q==0){
		for (i=0; i<mloc; i++){
		V(u, blockIndex_m+i, 0) = V(u, blockIndex_m+i, N);
		}
	}
	if(q==Q-1){
		for(i=0; i<mloc; i++){
		V(u, blockIndex_m+i, N+1) = V(u, blockIndex_m+i, 1);
		}
	}


}

// ... using 2D parallelization
void omp2dAdvect(int reps, double *u, int ldu) {
  int i, j;
  int r, ldv = N+2;
  double *v = calloc(ldv*(M+2), sizeof(double)); assert(v != NULL);
  printf("run 2d\n"); 
  /* 
  for (r = 0; r < reps; r++) {         
    for (j = 1; j < N+1; j++) { //top and bottom halo
      V(u, 0, j)   = V(u, M, j);
      V(u, M+1, j) = V(u, 1, j);
    }
    for (i = 0; i < M+2; i++) { //left and right sides of halo 
      V(u, i, 0) = V(u, i, N);
      V(u, i, N+1) = V(u, i, 1);
    }

    updateAdvectField(M, N, &V(u,1,1), ldu, &V(v,1,1), ldv);

    copyField(M, N, &V(v,1,1), ldv, &V(u,1,1), ldu);
  } //for (r...)
  */
  #pragma omp parallel default(shared) private(i,j,r)
  {
  int thread_i = omp_get_thread_num();
  int p = thread_i/Q;
  int q = thread_i%Q;
  int mloc = (M+2)/P;
  int nloc = (N+2)/Q;
  int bgp = p*mloc*(N+2)+q*nloc;
  mloc += (P-1)==p ? (M+2)%P : 0;
  nloc += (Q-1)==q ? (N+2)%Q : 0;
  int mloc_b = mloc, nloc_b = nloc, bgp_b = bgp;
  int local_work_load_i = mloc;
  int local_work_load_j = nloc;
  if(p == 0){local_work_load_i -= 1;bgp += N+2;}
  if(p == P-1){local_work_load_i -=1;}
  if(q == 0){local_work_load_j -=1;bgp += 1;}
  if(q==Q-1){local_work_load_j -= 1;}

  for (r = 0; r<reps; r++){
	omp2dBorderTB(u, ldu, bgp_b, mloc_b, nloc_b, p, q, P, Q);
	#pragma omp barrier
	omp2dBorderLR(u, ldu, bgp_b, mloc_b, nloc_b, p, q, P, Q);
	#pragma omp barrier
  	
	/*
	#pragma omp single
	{
	for (j = 1; j < N+1; j++) { //top and bottom halo
      	V(u, 0, j)   = V(u, M, j);
      	V(u, M+1, j) = V(u, 1, j);
    	}
    	for (i = 0; i < M+2; i++) { //left and right sides of halo 
      	V(u, i, 0) = V(u, i, N);
      	V(u, i, N+1) = V(u, i, 1);
	}	
	}//single
	*/

	updateAdvectField(local_work_load_i, local_work_load_j, &u[bgp], ldu, &v[bgp], ldv);
	#pragma omp barrier
	copyField(local_work_load_i,local_work_load_j,&v[bgp], ldv, &u[bgp],ldu);
	#pragma omp barrier
  }
  }
  free(v);
} //omp2dAdvect()


// ... extra optimization variant
void ompAdvectExtra(int reps, double *u, int ldu) {
  int i, j;
  int r, ldv = N+2;
  double *v = calloc(ldv*(M+2), sizeof(double)); assert(v != NULL);
  printf("run 2d\n"); 
  /* 
  for (r = 0; r < reps; r++) {         
    for (j = 1; j < N+1; j++) { //top and bottom halo
      V(u, 0, j)   = V(u, M, j);
      V(u, M+1, j) = V(u, 1, j);
    }
    for (i = 0; i < M+2; i++) { //left and right sides of halo 
      V(u, i, 0) = V(u, i, N);
      V(u, i, N+1) = V(u, i, 1);
    }

    updateAdvectField(M, N, &V(u,1,1), ldu, &V(v,1,1), ldv);

    copyField(M, N, &V(v,1,1), ldv, &V(u,1,1), ldu);
  } //for (r...)
  */
  #pragma omp parallel default(shared) private(i,j,r)
  {
  int thread_i = omp_get_thread_num();
  int p = thread_i/Q;
  int q = thread_i%Q;
  int mloc = (M+2)/P;
  int nloc = (N+2)/Q;
  int bgp = p*mloc*(N+2)+q*nloc;
  mloc += (P-1)==p ? (M+2)%P : 0;
  nloc += (Q-1)==q ? (N+2)%Q : 0;
  int mloc_b = mloc, nloc_b = nloc, bgp_b = bgp;
  int local_work_load_i = mloc;
  int local_work_load_j = nloc;
  if(p == 0){local_work_load_i -= 1;bgp += N+2;}
  if(p == P-1){local_work_load_i -=1;}
  if(q == 0){local_work_load_j -=1;bgp += 1;}
  if(q==Q-1){local_work_load_j -= 1;}
  double* readFrom;
  double* writeTo;

  for (r = 0; r<reps; r++){
	if(r%2 == 0){readFrom = u; writeTo = v;}
	if(r%2 != 0){readFrom = v; writeTo = u;}
	#pragma omp barrier
	omp2dBorderTB(readFrom, ldu, bgp_b, mloc_b, nloc_b, p, q, P, Q);
	#pragma omp barrier
	omp2dBorderLR(readFrom, ldu, bgp_b, mloc_b, nloc_b, p, q, P, Q);
	#pragma omp barrier
  	/*
	#pragma omp single
	{
	for (j = 1; j < N+1; j++) { //top and bottom halo
      	V(u, 0, j)   = V(u, M, j);
      	V(u, M+1, j) = V(u, 1, j);
    	}
    	for (i = 0; i < M+2; i++) { //left and right sides of halo 
      	V(u, i, 0) = V(u, i, N);
      	V(u, i, N+1) = V(u, i, 1);
	}	
	}//single
	*/
	updateAdvectField(local_work_load_i, local_work_load_j, &readFrom[bgp], ldu, &writeTo[bgp], ldv);
	#pragma omp barrier
  }
  if(reps%2 !=0){
	copyField(local_work_load_i,local_work_load_j,&v[bgp], ldv, &u[bgp],ldu);
	#pragma omp barrier
  }
  }
  free(v); 
} //ompAdvectExtra()
