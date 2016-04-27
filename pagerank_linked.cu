#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


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
    // compute the pagerank on the GPU

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


    // print the pagerank values computed on the GPU
    for (i=0;i<n_vertices;i++) {
        printf("AFTER GPU | Vertex %u:\tpagerank = %.6f\n", i, vertices[i].pagerank);
    }

/*****************************************************************************************/ 
  // Compute pagerank on host using old method for comparison purposes
    unsigned int i_iteration;

    float value, diff;
    float pr_dangling_factor = alpha / (float)n_vertices;   // pagerank to redistribute from dangling nodes
    float pr_dangling;
    float pr_random_factor = (1-alpha) / (float)n_vertices; // random portion of the pagerank
    float pr_random;
    float pr_sum, pr_sum_inv, pr_sum_dangling;
    float temp;

    // initialization of values before pagerank loop
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
    // NOTE: CAN PROBABLY REMOVE THIS SECTION FOR NOW

    /*size_t sizeOfVertices = n_vertices * sizeof(struct vertex);
    vertex *d_vertices = NULL;
    err = cudaMalloc((void **)&d_vertices, sizeOfVertices);
    err = cudaMemcpy(d_vertices, vertices, sizeOfVertices, cudaMemcpyHostToDevice);
    
    
    
    vertex ** d_testVar;
    cudaMalloc(&d_testVar, 3*sizeof(vertex*));
    cudaMemcpy(d_testVar, vertices[0].successors, 3*sizeof(vertex*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_vertices[0].successors),&d_testVar,sizeof(vertex**), cudaMemcpyHostToDevice);
    */


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
   // print the pageranks from this host computation
   for (i=0;i<n_vertices;i++) {
     printf("Vertex %u:\tpagerank = %.6f\n", i, vertices[i].pagerank);
   }
  /*************************************************************************/

    // Free device global memory
    // err = cudaFree(d_A);

    // Free host memory
    // free(h_A);

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

