/*
Simulation of flow inside a 3D Channel flow
using the lattice Boltzmann method (LBM)

Written by: Huiyang Yu (yu222@mail.ustc.edu.cn)       
Based on  : Abhijit Joshi (joshi1974@gmail.com)

Build instructions: nvcc -arch=sm_13 gpu_lbm.cu -o lbmGPU.x

Run instructions: ./lbmGPU.x
*/

#include<iostream>
#include<stdio.h>
#include<time.h>
#include<cuda.h>
#include<cuda_runtime.h>
//#include<chError.h>

// problem parameters
const int xDim = 128;
const int yDim = 128;
const int zDim = 128;
const int nDim = xDim*yDim*zDim;            
const int TIME_STEPS = 10000;  // number of time steps for which the simulation is run

const int NDIR = 19;           // number of discrete velocity directions used in the D2Q9 model

//const double DENSITY = 1.0;
//const double Re = 180;
//const double L=(zDim-1.0)/2.0;          // fluid density in lattice units
//const double tau=1.0;//0.5008;    
//const double nv=(2.0*tau-1.0)/6.0;
//const double u_tau=nv*Re/L;
//const double delta_tau=nv/u_tau;  
//const double Gravity = 0.0001;//u_tau*u_tau/L;
const double DENSITY = 1.0;          // fluid density in lattice units
const double LID_VELOCITY = 0.05;    // lid velocity in lattice units
const double REYNOLDS_NUMBER = 100;  // Re =
//----------------------------------------------
//v2.0
//----------------------------------------------
// calculate fluid viscosity based on the Reynolds number
const double kinematicViscosity = LID_VELOCITY * (double)xDim / REYNOLDS_NUMBER;

// calculate relaxation time tau
const double tau = 0.5 + 3.0 * kinematicViscosity;
__constant__ double _ex[NDIR] = {0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0};
__constant__ double _ey[NDIR] = {0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0};
__constant__ double _ez[NDIR] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
__constant__ double _alpha[NDIR] = {1.0 / 3.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
                                    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
                                    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
__constant__ int _ant[NDIR] = {0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15};
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

//__global__ void initialize(double *_ex, double *_ey, double *_ez, double *_alpha, int *_ant, double *_rh, double *_ux, double *_uy, double *_uz, double *_f, double *_f_new)
__global__ void initialize(double *_rh, double *_ux, double *_uy, double *_uz, double *_f, double *_f_new)
{

    // compute the "i", "j" and "k" location and the "dir"
    // handled by this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;
    int k = blockIdx.z * blockDim.z + threadIdx.z ;
    int ixy=i+j*(xDim)+k*(xDim*yDim);
    // initialize density and velocity fields inside the cavity
    // initializition with old flow field is looking forward to  
    _rh[ixy] = DENSITY;
    _ux[ixy] = 0;
    _uy[ixy] = 0;
    _uz[ixy] = 0;
    if(k==zDim-1) _ux[ixy] = LID_VELOCITY;
    // assign initial values for distribution functions   
    for(int dir=0;dir<NDIR;dir++) {
        int index =ixy+dir*nDim;
        double edotu = _ex[dir]*_ux[ixy] + _ey[dir]*_uy[ixy] + _ez[dir]*_uz[ixy];
        double udotu = _ux[ixy]*_ux[ixy] + _uy[ixy]*_uy[ixy] + _uz[ixy]*_uz[ixy];
        _f[index] = _rh[ixy] * _alpha[dir] * (1 + 3*edotu + 4.5*edotu*edotu - 1.5*udotu);
        //_f[index] = _feq[index];
        _f_new[index] = _f[index];
    }
}

//__global__ void timeIntegration(double *_ex, double *_ey, double *_ez, double *_alpha, int *_ant, double *_rh, double *_ux, double *_uy, double *_uz, double *_f, double *_f_new)
__global__ void timeIntegration(double *_rh, double *_ux, double *_uy, double *_uz, double *_f, double *_f_new)
{


    // compute the "i" and "j" location and the "dir"
    // handled by this thread

    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;
    int k = blockIdx.z * blockDim.z + threadIdx.z ;
    
    double _feq[NDIR];
    {
        // collision
        if((i>0) && (i<xDim-1) && (j>0) && (j<yDim-1) && (k>0) && (k<zDim-1)) 
        {
                int ixy=i+j*(xDim)+k*(xDim*yDim);
                for(int dir=0;dir<NDIR;dir++) 
                {
                    //int index =i+j*(xDim)+k*(xDim*yDim)+dir*nDim;
                    double edotu = _ex[dir]*_ux[ixy] + _ey[dir]*_uy[ixy] + _ez[dir]*_uz[ixy];
                    double udotu = _ux[ixy]*_ux[ixy] + _uy[ixy]*_uy[ixy] + _uz[ixy]*_uz[ixy];
                    _feq[dir] = _rh[ixy] * _alpha[dir] * (1 + 3*edotu + 4.5*edotu*edotu - 1.5*udotu);
                    
                }
        }
        if((i>0) && (i<xDim-1) && (j>0) && (j<yDim-1) && (k>0) && (k<zDim-1)) 
        {
            for(int dir=0;dir<NDIR;dir++)
            {
                int index =i+j*(xDim)+k*(xDim*yDim)+dir*nDim;
                int index_new = (i+_ex[dir])+(j+_ey[dir])*(xDim)+(k+_ez[dir])*(xDim*yDim)+dir*nDim;
                int index_ant = i+j*(xDim)+k*(xDim*yDim)+_ant[dir]*nDim;
                // post-collision distribution at (i,j) along "dir"
                double f_plus = _f[index] - (_f[index] - _feq[dir])/tau;//+3.0*_alpha[dir]*_ex[dir]*Gravity;
                if((i+_ex[dir]==0) || (i+_ex[dir]==xDim-1) || (j+_ey[dir]==0) || (j+_ey[dir]==yDim-1) || (k+_ez[dir]==0) || (k+_ez[dir]==zDim-1)) 
                {
                    // bounce back
                    int ixy = (i+_ex[dir])+(j+_ey[dir])*(xDim)+(k+_ez[dir])*(xDim*yDim);
                    double ubdote = _ux[ixy]*_ex[dir] + _uy[ixy]*_ey[dir];
                    _f_new[index_ant] = f_plus - 6.0 * DENSITY * _alpha[dir] * ubdote;
                }
                else 
                {
                    // stream to neighbor
                    _f_new[index_new] = f_plus;
                }
            }//for
        }//if

        // push f_new into f
        if((i>0) && (i<xDim-1) && (j>0) && (j<yDim-1) && (k>0) && (k<zDim-1)) 
        {
                double rho=0;
                for(int dir=0;dir<NDIR;dir++) 
                {
                    int index = i+j*(xDim)+k*(xDim*yDim)+dir*nDim;
                    _f[index] = _f_new[index];
                    rho += _f_new[index];
                }
                _rh[i + j * (xDim) + k * (xDim * yDim)] = rho;
        //}

        // update density at interior nodes
        /*if((i>0) && (i<xDim-1) && (j>0) && (j<yDim-1)&& (k>0) && (k<zDim-1)) {
                double rho=0;
                for(int dir=0;dir<NDIR;dir++) {
                    int index = i+j*(xDim)+k*(xDim*yDim)+dir*nDim;
                    rho+=_f_new[index];
                }
                _rh[i+j*(xDim)+k*(xDim*yDim)] = rho;
        }
        */
        // update velocity at interior nodes
        //if((i>0) && (i<xDim-1) && (j>0) && (j<yDim-1)&& (k>0) && (k<zDim-1)) 
        //{
                double velx=0;
                double vely=0;
                double velz=0;
                for(int dir=0;dir<NDIR;dir++) {
                    int index = i+j*(xDim)+k*(xDim*yDim)+dir*nDim;
                    velx+=_f_new[index]*_ex[dir];
                    vely+=_f_new[index]*_ey[dir];
                    velz+=_f_new[index]*_ez[dir];
                }
                //_ux[i+j*(xDim)+k*(xDim*yDim)] = velx/_rh[i+j*(xDim)+k*(xDim*yDim)];
                //_uy[i+j*(xDim)+k*(xDim*yDim)] = vely/_rh[i+j*(xDim)+k*(xDim*yDim)];
                //_uz[i+j*(xDim)+k*(xDim*yDim)] = velz/_rh[i+j*(xDim)+k*(xDim*yDim)];
                _ux[i + j * (xDim) + k * (xDim * yDim)] = velx / rho;
                _uy[i + j * (xDim) + k * (xDim * yDim)] = vely / rho;
                _uz[i + j * (xDim) + k * (xDim * yDim)] = velz / rho;
        }
    }

}

void timeOutput(int time, double *rho, double *ux, double *uy, double *uz) {
    int i,j,k;
    char FileName[8];
    int ixy;
    FILE *fp=NULL;
    sprintf(FileName,"F%d.dat",time);
    fp = fopen( FileName,"w");
    fprintf(fp,"Variables = x,y,z,rho,u,v,w\n");
    fprintf(fp,"ZONE I=%d,J=%d,K=%d,F=POINT\n",xDim,yDim,zDim);
    for(k=0;k<zDim;k++)
    {
        for(j=0;j<yDim;j++)
        {
            for(i=0;i<xDim;i++)
            {
                ixy=i+j*(xDim)+k*(xDim*yDim);
                fprintf(fp,"%d  %d   %d   %f   %f   %f   %f\n",i,j,k,rho[ixy],ux[ixy],uy[ixy],uz[ixy]);
            }
        }
    }
    fclose(fp);
    return;
}
int main(int argc, char *argv[])
{   
    time_t timep;
    int numDevices;
    cuInit(0);
    cuDeviceGetCount( &numDevices);
    printf("%d devices detected:\n",numDevices);
    for(int i = 0; i< numDevices;i++)
    {
        char szName[256];
        CUdevice device;
        cuDeviceGet(&device,i);
        cuDeviceGetName(szName,255,device);
        printf("\t%s\n",szName);
    }
    for(int i=0;i<numDevices;i++)
 {
  struct cudaDeviceProp device_prop;
  if(cudaGetDeviceProperties(&device_prop,i)==cudaSuccess)
  {
   printf("device properties is :\n"
      "\t device name is %s\n"
      "\t totalGlobalMem is %d\n"
      "\t sharedMemPerBlock is %d\n"
      "\t regsPerBlock is %d\n"
      "\t warpSize is %d\n"
      "\t memPitch is %d\n"
      "\t maxThreadsPerBlock is %d\n"
      "\t maxThreadsDim [3] is %d X %d X %d\n"
      "\t maxGridSize [3] is %d X %d X %d\n"
      "\t totalConstMem is %d\n"
      "\t device version is major %d ,minor %d\n"
      "\t clockRate is %d\n"
      "\t textureAlignment is %d\n"
      "\t deviceOverlap is %d\n"
      "\t multiProcessorCount is %d\n",
      device_prop.name,
      device_prop.totalGlobalMem,
      device_prop.sharedMemPerBlock,
      device_prop.regsPerBlock,
      device_prop.warpSize,
      device_prop.memPitch,
      device_prop.maxThreadsPerBlock,
      device_prop.maxThreadsDim[0],device_prop.maxThreadsDim[1],device_prop.maxThreadsDim[2],
      device_prop.maxGridSize[0],device_prop.maxGridSize[1],device_prop.maxGridSize[2],
      device_prop.totalConstMem,
      device_prop.major,device_prop.minor,
      device_prop.clockRate,
      device_prop.textureAlignment,
      device_prop.deviceOverlap,
      device_prop.multiProcessorCount);
  }
 }
/*
    // the base vectors and associated weight coefficients (GPU)
    double *_ex, *_ey, *_ez, *_alpha;  // pointers to device (GPU) memory
    cudaMalloc((void **)&_ex,NDIR*sizeof(double));
    cudaCheckErrors("cudaMalloc _ex fail");
    cudaMalloc((void **)&_ey,NDIR*sizeof(double));
    cudaCheckErrors("cudaMalloc _ey fail");
    cudaMalloc((void **)&_ez,NDIR*sizeof(double));
    cudaCheckErrors("cudaMalloc _ez fail");
    cudaMalloc((void **)&_alpha,NDIR*sizeof(double));
    cudaCheckErrors("cudaMalloc _alpha fail");

    // ant vector (GPU)
    int *_ant;  // gpu memory
    cudaMalloc((void **)&_ant,NDIR*sizeof(int));
    cudaCheckErrors("cudaMalloc _ant fail");
*/
    // allocate memory on the GPU
    double *_f,  *_f_new;//*_feq,
    cudaMalloc((void **)&_f,xDim*yDim*zDim*NDIR*sizeof(double));
    cudaCheckErrors("cudaMalloc _f fail");
    cudaMalloc((void **)&_f_new,xDim*yDim*zDim*NDIR*sizeof(double));
    cudaCheckErrors("cudaMalloc _f_new fail");

    double *_rh, *_ux, *_uy, *_uz;
    cudaMalloc((void **)&_rh,xDim*yDim*zDim*sizeof(double));
    cudaCheckErrors("cudaMalloc _rh fail");
    cudaMalloc((void **)&_ux,xDim*yDim*zDim*sizeof(double));
    cudaCheckErrors("cudaMalloc _ux fail");
    cudaMalloc((void **)&_uy,xDim*yDim*zDim*sizeof(double));
    cudaCheckErrors("cudaMalloc _uy fail");
    cudaMalloc((void **)&_uz,xDim*yDim*zDim*sizeof(double));

    double *rh_host, *ux_host, *uy_host, *uz_host;
    rh_host=(double *)malloc(xDim*yDim*zDim*sizeof(double));
    ux_host=(double *)malloc(xDim*yDim*zDim*sizeof(double));
    uy_host=(double *)malloc(xDim*yDim*zDim*sizeof(double));
    uz_host=(double *)malloc(xDim*yDim*zDim*sizeof(double));
    //cudaGetDeviceGetProperties();
    //int thr_num=deviceProp.maxThreadsPerBlock;
    //printf("thr_num=%d\n",thr_num);
    // assign a 3D distribution of CUDA "threads" within each CUDA "block"    
    int threadsAlongX=32, threadsAlongY=32,threadsAlongZ=1;
    dim3 dimBlock(threadsAlongX, threadsAlongY, threadsAlongZ);

    // calculate number of blocks along X and Y in a 2D CUDA "grid"
    dim3 dimGrid( int(float(xDim)/float(dimBlock.x)), int(float(yDim)/float(dimBlock.y)), int(float(zDim)/float(dimBlock.z)));

    initialize<<<dimGrid,dimBlock>>>(_rh, _ux, _uy, _uz, _f, _f_new);
    cudaCheckErrors("initialize fail");
    cudaDeviceSynchronize();
    // time integration
    int time_s=0;
    cudaMemcpy(rh_host, _rh,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ux_host, _ux,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(uy_host, _uy,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(uz_host, _uz,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
    if(rh_host[1]==0.0) {return 0;}
    timeOutput(time_s,rh_host,ux_host,uy_host,uz_host);
    time(&timep);
    printf("%s",ctime(&timep));
    while(time_s<TIME_STEPS) {

        time_s++;

        timeIntegration<<<dimGrid,dimBlock >>>(_rh, _ux, _uy, _uz, _f, _f_new);
        cudaCheckErrors("initialize fail");
        cudaDeviceSynchronize();

        /*if(time_s%(TIME_STEPS/10)==0) {
             cudaMemcpyAsync(rh_host, _rh,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
             cudaMemcpyAsync(ux_host, _ux,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
             cudaMemcpyAsync(uy_host, _uy,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
             cudaMemcpyAsync(uz_host, _uz,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
             timeOutput(time_s,rh_host,ux_host,uy_host,uz_host);
        }*/
        
    }
    time (&timep);
    printf("%s",ctime(&timep));
    cudaMemcpy(rh_host, _rh,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ux_host, _ux,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(uy_host, _uy,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(uz_host, _uz,xDim*yDim*zDim*sizeof(double), cudaMemcpyDeviceToHost);
    timeOutput(time_s,rh_host,ux_host,uy_host,uz_host);
    cudaFree(_ex);
    cudaFree(_ey);
    cudaFree(_ez);
    cudaFree(_alpha);
    cudaFree(_ant);
    cudaFree(_f);
    //cudaFree(_feq);
    cudaFree(_f_new);
    cudaFree(_rh);
    cudaFree(_ux);
    cudaFree(_uy);
    cudaFree(_uz);

    return 0;
}
