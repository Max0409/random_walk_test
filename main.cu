#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <gsl/gsl_rng.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand_mtgp32.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <curand.h>
using namespace std;
const int MAX_STRING=100;
const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;
class ClassVertex
{
public:
    char *name;
};

class HashTable;

class OriginalGraph{
private:
    int num_vertices = 0;
    long long num_edges = 0;
    int *edge_source_id;
    int *edge_target_id;
    int *weight;
    float vertex_sum_degrees = 0.0;
    
    int *alias;
    float *prob;
    int *flat_offsets;
    int *neighbor_ids;
    float *neighbor_weights;
    
    int *d_alias;
    float *d_prob;
    int *d_flat_offsets;
    int *d_neighbor_ids;
    float *d_neighbor_weights;
 
public:
    int GetNumVertices();
    void NumVerticesAddOne();
    void ConstructCSR();
    void LoadData(char*  train_file,HashTable* hash_table);
    void InitNeighborIds();
    void BuildNeighborAliasTable();
    void InitializeProbs(int count,int *alias,float *prob,float *weights);
    int SampleAVertice(int* alias,float *prob,double rand_value1, double rand_value2,int offset_u,int offset_u_plus1);
    void RandomWalkWithCPU(int random_walk_length);
    void RandomWalkWithGPU();
    void CopyDataToHost();
    void CopyDataToDevice();
};


class HashTable
{
private:
    ClassVertex *vertex;
    int *vertex_hash_table;
    int hash_table_size ;
    int max_num_vertices = 1000;
public:
    ~HashTable();
    unsigned int Hash(char *key);
    void InitHashTable(int hash_table_size);
    void InsertHashTable(char *key, int value);
    int SearchHashTable(char *key);
    int AddVertex(char *name,OriginalGraph* original_graph);

};
HashTable::~HashTable()
{
   delete[] vertex;
   delete[] vertex_hash_table;
}


/* Build a hash table, mapping each vertex name to a unique vertex id */
unsigned int HashTable:: Hash(char *key)
{
    unsigned int seed = 131;
    unsigned int hash = 0;
    while (*key)
    {
hash = hash * seed + (*key++);
    }
    return hash % hash_table_size;
}

void HashTable::InitHashTable(int hash_table_size)
{
    this->hash_table_size=hash_table_size;
    vertex_hash_table = (int *)malloc(this->hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) vertex_hash_table[k] = -1;
    vertex = (class ClassVertex *)calloc(max_num_vertices, sizeof(class ClassVertex));
}

void HashTable::InsertHashTable(char *key, int value)
{
    int addr = Hash(key);
    while (vertex_hash_table[addr] != -1) addr = (addr + 1) % hash_table_size;
    vertex_hash_table[addr] = value;
}

int HashTable::SearchHashTable(char *key)
{
    int addr = Hash(key);
    while (1)
    {
        if (vertex_hash_table[addr] == -1) return -1;
        if (!strcmp(key, vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr];
        addr = (addr + 1) % hash_table_size;
    }
    return -1;
}

/* Add a vertex to the vertex set */
int HashTable::AddVertex(char *name,OriginalGraph* original_graph)
{
    int length = strlen(name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vertex[original_graph->GetNumVertices()].name = (char *)calloc(length, sizeof(char));
    strcpy(vertex[original_graph->GetNumVertices()].name, name);
    original_graph->NumVerticesAddOne();
    if (original_graph->GetNumVertices() + 2 >= max_num_vertices)
    {
        max_num_vertices += 1000;
        vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
    }
    InsertHashTable(name, original_graph->GetNumVertices() - 1);
    return original_graph->GetNumVertices() - 1;
}



int OriginalGraph::GetNumVertices()
{
    return this->num_vertices;
}
void OriginalGraph::NumVerticesAddOne()
{
    this->num_vertices++;
}
void OriginalGraph::LoadData(char*  train_file,HashTable* hash_table)
{
    FILE *fin;
    char name_v1[MAX_STRING], name_v2[MAX_STRING];
    char str[2 * MAX_STRING + 10000];
    int vid, u, v;
    int nb;
    fin = fopen(train_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: network file not found!\n");
        exit(1);
    }
    this->num_edges = 0;
    while (fgets(str, sizeof(str), fin)) this->num_edges++;
    fclose(fin);
    printf("Number of edges: %lld          \n", this->num_edges);
    this->edge_source_id = (int *)malloc(this->num_edges*sizeof(int));
    this->edge_target_id = (int *)malloc(this->num_edges*sizeof(int));
    this->weight = (int *)malloc(this->num_edges*sizeof(int));
    fin = fopen(train_file, "rb");
    this->num_vertices = 0;
    double weight;
    for (int k = 0; k != this->num_edges; k++)
    {
        fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);
        if (k % 10000 == 0)
        {
            printf("Reading edges: %.3lf%%%c", k / (double)(this->num_edges + 1) * 100, 13);
            fflush(stdout);
        }
        vid = hash_table->SearchHashTable(name_v1);
        if (vid == -1) vid = hash_table->AddVertex(name_v1,this);
        this->edge_source_id[k] = vid;
        vid = hash_table->SearchHashTable(name_v2);
        if (vid == -1) vid = hash_table->AddVertex(name_v2,this);
        this->edge_target_id[k] = vid;
        this->weight[k] = (int)weight;
    }
    fclose(fin);
    printf("Number of vertices: %d          \n", this->num_vertices);
    printf("\n");
}

void OriginalGraph::InitNeighborIds()
{
    flat_offsets = (int*)malloc((num_vertices + 1) * sizeof(int));
    for(long long i=0;i<num_vertices+1;i++)
    {
        flat_offsets[i]=0;
        
    }
    neighbor_ids = (int *)malloc(num_edges*sizeof(int));
    for(long long i=0;i<num_edges;i++)
    {
        neighbor_ids[i]=-1;
        
    }
    neighbor_weights = (float *)malloc(num_edges*sizeof(float));
    for(long long i=0;i<num_edges;i++)
    {
        neighbor_weights[i]=0;
        
    }
    for (long long k = 0; k != num_edges; k++)
    {
           
           int h = edge_source_id[k] + 1;
           flat_offsets[h]++;
    }
    for(int i = 1; i < num_vertices + 1; i++)
    {
        flat_offsets[i] += flat_offsets[i - 1];
    }
    
    vertex_sum_degrees = 0.0;
    for (long long k = 0; k != num_edges; k++)
    {
        
        int index = flat_offsets[edge_source_id[k]];
        while(neighbor_ids[index] != -1)
        {
            index++;
        }
        neighbor_ids[index] = edge_target_id[k];
        neighbor_weights[index] = weight[k];
        vertex_sum_degrees += weight[k] * 2;
    }
    printf("InitNeighborIds with %d vertices and %lld edges from OriginalGraph",
            num_vertices, num_edges);
}

void OriginalGraph::BuildNeighborAliasTable()
{
    alias = (int *)malloc(num_edges*sizeof(int));
    prob = (float *)malloc(num_edges*sizeof(float));
    for(int k = 0; k < num_vertices; k++)
    {
        int offset = flat_offsets[k];
        int count = flat_offsets[k + 1] - offset;
        InitializeProbs(count, alias + offset,
            prob + offset, neighbor_weights + offset);
    }
    
}

void OriginalGraph::InitializeProbs(int count,int *alias,float *prob,float *weights)
{
    if (count == 0)
        return;

    float *norm_prob = (float *)malloc(count*sizeof(float));
    int *large_block = (int*)malloc(count*sizeof(int));
    int *small_block = (int*)malloc(count*sizeof(int));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    float sum = 0;
    int cur_small_block, cur_large_block;
    int num_small_block = 0, num_large_block = 0;
    for (int k = 0; k != count; k++) sum += weights[k];
    float average = count / sum;
    for (int k = 0; k != count; k++) norm_prob[k] = weights[k] * average;
    for (int k = count - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }
    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        prob[cur_small_block] = norm_prob[cur_small_block];
        alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }
    while (num_large_block) prob[large_block[--num_large_block]] = 1;
    while (num_small_block) prob[small_block[--num_small_block]] = 1;
    free(norm_prob);
    free(small_block);
    free(large_block);
    
}

int OriginalGraph::SampleAVertice(int* alias,float *prob,double rand_value1, double rand_value2,int offset_u,int offset_u_plus1)
{
    int count = offset_u_plus1 - offset_u;
    int k = (int)count * rand_value1;
    return rand_value2 < prob[k+offset_u] ? k : alias[k+offset_u];
}

void OriginalGraph::RandomWalkWithCPU(int random_walk_length)
{
    int u,v;
    
    int *head_chains = (int *)malloc(random_walk_length * sizeof(int));
    int *tail_chains = (int *)malloc(random_walk_length * sizeof(int));
 
    for(int i=0;i<num_vertices;i++)
    {
        u=i;
        int sample_length=0;
        int index;
        if(flat_offsets[u + 1] -flat_offsets[u])
        {
            index = flat_offsets[u] + SampleAVertice(alias,prob,gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r),flat_offsets[u],flat_offsets[u+1]);
            v = neighbor_ids[index];
            head_chains[0] = u;
            tail_chains[0] = v;
            sample_length++;
            for(int i = 1; i < random_walk_length; i++)
            {
                if(flat_offsets[u + 1] - flat_offsets[u])
                {
                    u = v;
                    index = flat_offsets[u] + SampleAVertice(alias,prob,gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r),flat_offsets[u],flat_offsets[u+1]);
                    v = neighbor_ids[index];
                    head_chains[i] = u;
                    tail_chains[i] = v;
                    sample_length++;
                }
                else
                    break;
            }
        }
    }
}
__device__ int SampleAVertice(int *alias,
                float *prob,
                double rand_value1,
                double rand_value2,
                int offset_u,
                int offset_u_plus1)
{
        int count = offset_u_plus1 - offset_u;
        int k = (int)count * rand_value1;
        return rand_value2 < prob[k+offset_u] ? k : alias[k+offset_u];
}

__global__ void RandomWalkKernal(int num_vertices,
                int num_edges,
                int *flat_offsets,
                int *neighbor_ids,
                float *neighbor_weights,
                int *alias,
                float *prob,
                long seed
               )
{

    int u = blockDim.x *  blockIdx.x + threadIdx.x;
    if (u >= num_vertices)
    {
        return;
    };
    curandState state;
    curand_init(seed, u, 0, &state);
    int v;
    int head_chains[20];
    int tail_chains[20];
    int sample_length = 0;
    int index;
    if (flat_offsets[u+1] - flat_offsets[u])
    {
        double rand_value1 = curand_uniform_double(&state);
        double rand_value2 = curand_uniform_double(&state);

        index = flat_offsets[u] + SampleAVertice(
                        alias,
                        prob,
                        rand_value1,
                        rand_value2,
                        flat_offsets[u],
                        flat_offsets[u+1]);

        v = neighbor_ids[index];
        head_chains[0] = u;
        tail_chains[0] = v;
        sample_length++;
        for(int i = 1; i < random_walk_length; i++)
        {
            if(flat_offsets[u + 1] - flat_offsets[u])
            {
                u = v;
                double rand_value1 = curand_uniform_double(&state);
                double rand_value2 = curand_uniform_double(&state);
                index = flat_offsets[u] + SampleAVertice(
                                alias,
                                prob,
                                rand_value1,
                                rand_value2,
                                flat_offsets[u],
                                flat_offsets[u+1]);

                v = neighbor_ids[index];
                head_chains[i] = u;
                tail_chains[i] = v;
                sample_length++;
            }
            else
            {
                 break;
            }
        }
    }
}


void OriginalGraph::RandomWalkWithGPU()
{
    CopyDataToDevice();
    long seed = unsigned(time(NULL));
    int num_threads = 256;
    int num_blocks = (device_data.num_vertices + num_threads - 1) / num_threads;
    trainSamples<<<num_blocks, num_threads>>> (
                           num_vertices,
                           num_edges,
                           d_flat_offsets,
                           d_neighbor_ids,
                           d_neighbor_weights,
                           d_alias,
                           d_prob,
                           seed,
                           );
    
    cudaDeviceSynchronize();
    CopyDataToHost();
    
}


void OriginalGraph::CopyDataToHost()
{
    cudaFree(d_flat_offsets);
    cudaFree(d_neighbor_ids);
    cudaFree(d_neighbor_weights);
    cudaFree(d_alias);
    cudaFree(d_prob);
}
void OriginalGraph::CopyDataToDevice()

{
    cudaError_t err = cudaMalloc((void **)&d_flat_offsets,
                      (num_vertices + 1) *  sizeof(int));

    if (err != cudaSuccess)
    {
        printf("Error: memory allocation for d_flat_offsets failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_flat_offsets,
                     flat_offsets,
                     (num_vertices + 1) * sizeof(int),
                     cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        printf("Error: memory copy for d_flat_offsets failed!\n");
        exit(1);
    }


    err = cudaMalloc((void **)&d_neighbor_ids,
                    num_edges * sizeof(int));

    if (err != cudaSuccess)
    {
        printf("Error: memory allocation for d_neighbor_ids failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_neighbor_ids,
                     neighbor_ids,
                     num_edges * sizeof(int),
                     cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        printf("Error: memory copy of d_neighbor_ids failed!\n");
        exit(1);
    }
    

    err = cudaMalloc((void **)&d_neighbor_weights,
                     num_edges * sizeof(float));

    if (err != cudaSuccess)
    {
        printf("Error: memory allocation for d_neighbor_weights failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_neighbor_weights,
                     neighbor_weights,
                     num_edges * sizeof(float),
                     cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        printf("Error: memory copy of d_neighbor_weights failed!\n");
        exit(1);
    }
    
    err = cudaMalloc((void **)&d_alias, num_edges * sizeof(int));

    if (err != cudaSuccess)
    {
        printf("Error: memory allocation for d_alias failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_alias,
                     alias,
                     num_edges * sizeof(int),
                     cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        printf("Error: memory copy of d_alias failed!\n");
        exit(1);
    }

    err = cudaMalloc((void **)&d_prob, num_edges * sizeof(float));

    if (err != cudaSuccess)
    {
        printf("Error: memory allocation for d_prob failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_prob,
                     h_prob,
                     num_edges * sizeof(float),
                     cudaMemcpyHostToDevice);

    if(err != cudaSuccess)
    {
        printf("Error: memory copy of d_prob failed!\n");
        exit(1);
    }
  
    
}

int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}


int main(int argc, char **argv) {
   
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);
    int hash_table_size;
    char train_file[MAX_STRING];
    int random_walk_length;
    int i;
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-hash_table_size", argc, argv)) > 0) hash_table_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-random_walk_length", argc, argv)) > 0) random_walk_length = atoi(argv[i + 1]);
     HashTable hash_table;
    OriginalGraph original_graph;
    hash_table.InitHashTable(hash_table_size);
    original_graph.LoadData(train_file,&hash_table);
    original_graph.InitNeighborIds();
    original_graph.BuildNeighborAliasTable();
    printf("BuildNeighborAliasTable\n");
    //original_graph.RandomWalkWithCPU(random_walk_length);
    //printf("RandomWalkWithCPU\n");
    original_graph.RandomWalkWithGPU();
    printf("RandomWalkWithGPU\n");
    
    return 0;
}
