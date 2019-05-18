#include <ctime>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace std;

// static int *result;

void display(vector<int> &mat)
{
    int size = mat.size();
    int rows = (int)sqrt(size);

    cout << "Output Matrix:" << endl;

    for (int i = 0; i < size; i++)
    {
        cout << mat[i] << " ";

        if ((i + 1) % rows == 0)
        {
            cout << endl;
        }
    }
}

void display(int mat[], int size)
{

    cout << "Output Matrix:" << endl;

    for (int i = 0; i < size * size; i++)
    {
        cout << mat[i] << " ";

        if ((i + 1) % size == 0)
        {
            cout << endl;
        }
    }
}

void matMulProcess(int result[], vector<int> &mat1, vector<int> &mat2, int process_number)
{
    int size = mat1.size();
    int rows = (int)sqrt(size);

    for (int j = 0; j < rows; j++)
    {
        int temp_value = 0;

        for (int k = 0; k < rows; k++)
        {
            temp_value += mat1[process_number * rows + k] * mat2[k * rows + j];
        }

        result[process_number * rows + j] = temp_value;
    }
}

void fillMat(vector<int> &mat)
{
    int size = mat.size();

    for (int i = 0; i < size; i++)
    {
        mat[i] = (rand() % 20) + 1;
    }
}

int main(int argc, char *argv[])
{
    // TODO get shape from params
    srand(time(0));
    unsigned int size = atoi(argv[1]);
    vector<int> mat1(size * size);
    vector<int> mat2(size * size);
    int *result = (int *)mmap(NULL, size * size * sizeof(int), PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    // Fill matrix with random numbers
    fillMat(mat1);
    fillMat(mat2);
    pid_t child_process[size];
    int pid_status[size];
    auto start = std::chrono::high_resolution_clock::now();

    // Create one process from every mat column.
    for (unsigned int i = 0; i < size; i++)
    {
        pid_t pid_process = fork();

        if (pid_process == 0)
        { // on child process
            // cout << "child " << i << endl;
            matMulProcess(result, mat1, mat2, i);
            exit(EXIT_SUCCESS);
        }
    }
    for (int t = 0; t < size; t++)
    {
        waitpid(child_process[t], &pid_status[t], 0);
    }
    // display(mat1);
    // display(mat2);
    // display(result, size);
    // display(result);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = finish - start;

    // display(result);

    cout << size << " : " << elapsed.count() << endl;

    return 0;
}