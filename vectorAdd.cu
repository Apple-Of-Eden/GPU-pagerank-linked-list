/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}



typedef struct vertex vertex;

struct vertex {
    unsigned int vertex_id;
    float pagerank;
    float pagerank_next;
    unsigned int n_successors;
    vertex ** successors;
};

float abs_float(float in) {
  if (in >= 0)
    return in;
  else
    return -in;
}


__global__ void setPagerankNextToZero(vertex * vertices) {
    int i = threadIdx.x;
    
    vertices[i].pagerank_next = 0;
}

__global__ void initializePageranks(vertex * vertices, int n_vertices) {
    int i = threadIdx.x;
    
    vertices[i].pagerank = 1.0/(float)n_vertices;
}


__global__ void addToNextPagerank(vertex * vertices, float * dangling_value) {
    int i = threadIdx.x;
    int j;

    if(vertices[i].n_successors > 0) {
        for(j = 0; j < vertices[i].n_successors; j++) {
            atomicAdd(&(vertices[i].successors[j]->pagerank_next), 0.85*(vertices[i].pagerank)/vertices[i].n_successors);
        }
    }else {
        atomicAdd(dangling_value, 0.85*vertices[i].pagerank);
    }
}

__global__ void finalPagerankForIteration(vertex * vertices, int n_vertices, float dangling_value){
    int i = threadIdx.x;

    vertices[i].pagerank_next += (dangling_value + (1-0.85))/((float)n_vertices);
    
}

__global__ void setPageranksFromNext(vertex * vertices) {
    int i = threadIdx.x;

    vertices[i].pagerank = vertices[i].pagerank_next;
}


int main(void) {
// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

/*************************************************************************/
  // build up the graph
  int i,j;
  unsigned int n_vertices = 0;
  unsigned int n_edges = 0;
  unsigned int vertex_from = 0, vertex_to = 0;

  vertex * vertices;

  FILE * fp;
  if ((fp = fopen("testInput.txt", "r")) == NULL) {
    fprintf(stderr,"ERROR: Could not open input file.\n");
    exit(-1);
  }


  // parse input file to count the number of vertices
  // expected format: vertex_from vertex_to
  while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
    if (vertex_from > n_vertices)
      n_vertices = vertex_from;
    else if (vertex_to > n_vertices)
      n_vertices = vertex_to;
  }
  n_vertices++;
    
// CALC NUMBER OF OUTGOING LINKS PER PAGE ***********************************
    unsigned int * outgoingLinks = (unsigned int *) calloc(n_vertices,sizeof(unsigned int)); 
    fseek(fp,0L, SEEK_SET);
    while(fscanf(fp,"%u %u", &vertex_from, &vertex_to) != EOF) {
        outgoingLinks[vertex_from] += 1;
    }

  // allocate memory for vertices
 //vertices = (vertex *)malloc(n_vertices*sizeof(vertex));
  err = cudaMallocManaged((void **)&vertices, n_vertices*sizeof(vertex));
 // err = cudaMemcpy(d_vertices, vertices, sizeOfVertices, cudaMemcpyHostToDevice);

    // SET Initial values  **********************************************************
    unsigned int n_iterations = 25;
    float alpha = 0.85;
    float eps   = 0.000001;
   

 if (!vertices) {
    fprintf(stderr,"Malloc failed for vertices.\n");
    exit(-1);
  }
  memset((void *)vertices, 0, (size_t)(n_vertices*sizeof(vertex)));

  // parse input file to count the number of successors of each vertex
  fseek(fp, 0L, SEEK_SET);
  while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
    vertices[vertex_from].n_successors++;
    n_edges++;
  }

  // allocate memory for successor pointers
  for (i=0; i<n_vertices; i++) {
    vertices[i].vertex_id = i;
    if (vertices[i].n_successors > 0) {
     // vertices[i].successors = (vertex **)malloc(vertices[i].n_successors*sizeof(vertex *));
      err = cudaMallocManaged((void***)&vertices[i].successors,vertices[i].n_successors*sizeof(vertex*));
      if (!vertices[i].successors) {
        fprintf(stderr,"Malloc failed for successors of vertex %d.\n",i);
        exit(-1);
      }
      memset((void *)vertices[i].successors, 0, (size_t)(vertices[i].n_successors*sizeof(vertex *)));
    }
    else
      vertices[i].successors = NULL;
  }

  // parse input file to set up the successor pointers
  fseek(fp, 0L, SEEK_SET);
  while (fscanf(fp, "%d %d", &vertex_from, &vertex_to) != EOF) {
    for (i=0; i<vertices[vertex_from].n_successors; i++) {
      if (vertices[vertex_from].successors[i] == NULL) {
        vertices[vertex_from].successors[i] = &vertices[vertex_to];
        break;
      }
      else if (i==vertices[vertex_from].n_successors-1) {
        printf("Setting up the successor pointers of virtex %u failed",vertex_from);
        return -1;
      }
    }
  }

  fclose(fp);

// PRINT THE DATASTRUCTURE
   /* for(i = 0; i < n_vertices; i++) {
        printf("Page: %d, Suc: ", (vertices+i)->vertex_id);
        
        
        for(j = 0; j < (vertices+i)->n_successors; j++) {
            printf("%d, ",(vertices+i)->successors[j]->vertex_id);
        }
        printf("\n");
    }*/   

  /*************************************************************************/
  // compute the pagerank

    float dangling_value_h = 0;
    float * dangling_value_d;
    
    err = cudaMalloc((void **)&dangling_value_d, sizeof(float));
    err = cudaMemcpy(dangling_value_d, &dangling_value_h, sizeof(float), cudaMemcpyHostToDevice);
    //err = cudaMallocManaged((void *)&dangling_value, sizeof(float));

    initializePageranks<<<1,46>>>(vertices, n_vertices);
   cudaDeviceSynchronize();
    
    for(i = 0; i < 23; i++) {
        // set the next pagerank values to 0
        setPagerankNextToZero<<<1,46>>>(vertices);
        cudaDeviceSynchronize();
       
        // set the dangling value to 0 
        dangling_value_h = 0;
        err = cudaMemcpy(dangling_value_d, &dangling_value_h, sizeof(float), cudaMemcpyHostToDevice);
        
        // initial parallel pagerank_next computation
        addToNextPagerank<<<1,46>>>(vertices, dangling_value_d);
        cudaDeviceSynchronize();

        // get the dangling value
        err = cudaMemcpy(&dangling_value_h, dangling_value_d, sizeof(float), cudaMemcpyDeviceToHost);
        printf("the dangling_value is now: %.3f\n",dangling_value_h);
 
        // final parallel pagerank_next computation
        finalPagerankForIteration<<<1,46>>>(vertices, n_vertices, dangling_value_h);
         cudaDeviceSynchronize();

        setPageranksFromNext<<<1,46>>>(vertices);
        cudaDeviceSynchronize(); 
    }

 /*   float dangling_value = 0;
    int q;

    for (i=0; i < 23; i++) {
        for(j=0; j < n_vertices; j++) {
            vertices[j].pagerank_next = 0;
        }

        dangling_value = 0;

        for(j=0; j < n_vertices; j++) {
            if(vertices[j].successors > 0) {
               for(q=0; q < vertices[j].successors; q++) {
                    vertices[j].successors[q]->pagerank_next += (0.85*vertices[j].pagerank)/(float)vertices[j].successors; 
                }        
            } else {
                dangling_value += 0.85 * vertices[j].pagerank;
            }
        }

        for(j = 0; j < n_vertices; j++) {
            vertices[j].pagerank_next = 
        }

    }*/


    // print
    for (i=0;i<n_vertices;i++) {
        printf("AFTER GPU | Vertex %u:\tpagerank = %.6f\n", i, vertices[i].pagerank);
    }

 
  // run on the host
  unsigned int i_iteration;

   float value, diff;
   float pr_dangling_factor = alpha / (float)n_vertices;   // pagerank to redistribute from dangling nodes
   float pr_dangling;
   float pr_random_factor = (1-alpha) / (float)n_vertices; // random portion of the pagerank
   float pr_random;
   float pr_sum, pr_sum_inv, pr_sum_dangling;
   float temp;

   // initialization
   for (i=0;i<n_vertices;i++) {
     vertices[i].pagerank = 1 / (float)n_vertices;
     vertices[i].pagerank_next =  0;
   }

   pr_sum = 0;
   pr_sum_dangling = 0;
   for (i=0; i<n_vertices; i++) {
     pr_sum += vertices[i].pagerank;
     if (!vertices[i].n_successors)
       pr_sum_dangling += vertices[i].pagerank;
   }

   i_iteration = 0;
   diff = eps+1;
//****** transfer data structure to CUDA memory ************************************************ 
    /*size_t sizeOfVertices = n_vertices * sizeof(struct vertex);
    vertex *d_vertices = NULL;
    err = cudaMalloc((void **)&d_vertices, sizeOfVertices);
    err = cudaMemcpy(d_vertices, vertices, sizeOfVertices, cudaMemcpyHostToDevice);
    
    
    
    vertex ** d_testVar;
    cudaMalloc(&d_testVar, 3*sizeof(vertex*));
    cudaMemcpy(d_testVar, vertices[0].successors, 3*sizeof(vertex*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_vertices[0].successors),&d_testVar,sizeof(vertex**), cudaMemcpyHostToDevice);
    */


   // testFunction<<<1,1>>>(vertices);
   // cudaDeviceSynchronize();
  //  printf("for comp: %p\n", vertices[0].successors);

//********************************************************************************************* 
   while ( (diff > eps) && (i_iteration < n_iterations) ) {

     for (i=0;i<n_vertices;i++) {
       if (vertices[i].n_successors)
            value = (alpha/vertices[i].n_successors)*vertices[i].pagerank;  //value = vote value after splitting equally
       else
            value = 0;
       //printf("vertex %d: value = %.6f \n",i,value);
       for (j=0;j<vertices[i].n_successors;j++) {               // pagerank_next = sum of votes linking to it
            vertices[i].successors[j]->pagerank_next += value;
       }
     }
    
    // for normalization
     pr_sum_inv = 1/pr_sum;

     // alpha
     pr_dangling = pr_dangling_factor * pr_sum_dangling;
     pr_random = pr_random_factor * pr_sum;

     pr_sum = 0;
     pr_sum_dangling = 0;

     diff = 0;
     for (i=0;i<n_vertices;i++) {

       // update pagerank
       temp = vertices[i].pagerank;
       vertices[i].pagerank = vertices[i].pagerank_next*pr_sum_inv + pr_dangling + pr_random;
       vertices[i].pagerank_next = 0;

       // for normalization in next cycle
       pr_sum += vertices[i].pagerank;
       if (!vertices[i].n_successors)
            pr_sum_dangling += vertices[i].pagerank;

       // convergence
       diff += abs_float(temp - vertices[i].pagerank);
     }
     printf("Iteration %u:\t diff = %.12f\n", i_iteration, diff);

     i_iteration++;
   }

 /*************************************************************************/
   // print
   for (i=0;i<n_vertices;i++) {
     printf("Vertex %u:\tpagerank = %.6f\n", i, vertices[i].pagerank);
   }
  /*************************************************************************/

 
    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

