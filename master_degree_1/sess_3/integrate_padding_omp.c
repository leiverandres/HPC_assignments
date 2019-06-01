#include <stdio.h>
#include <omp.h>

static long num_steps = 1000000000;
double step;

#define NUM_THREADS 2
#define PAD 8

int main()
{
        int i, nThreads;
        double pi, sum[NUM_THREADS][PAD];

        step = 1.0/(double) num_steps;

        omp_set_num_threads(NUM_THREADS);
        #pragma omp parallel
        {
        int i, ID, nThreadsInternal;
        double x;
        ID = omp_get_thread_num();
        nThreadsInternal = omp_get_num_threads();
        if(ID == 0) nThreads = nThreadsInternal;
        for( i=ID, sum[ID][0]=0.0; i<num_steps; i=i+nThreadsInternal) {
                x = (i+0.5)*step;
                sum[ID][0] += 4.0/(1.0 + x*x);
        }

        }

        for (i=0, pi=0.0; i<nThreads; i++)
                pi += sum[i][0]*step;

        printf("%f\n", pi);
}
