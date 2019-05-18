#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <thread>

using namespace std;

void display(const vector<vector<int>> &mat)
{
    int size = mat.size();
    cout << "Output Matrix:" << endl;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
}

void thread_dot(vector<vector<int>> &result, vector<vector<int>> &mat1, vector<vector<int>> &mat2, int index)
{
    int size = mat1.size();
    for (int j = 0; j < size; j++)
    {
        for (int k = 0; k < size; k++)
        {
            result[index][j] += mat1[index][k] * mat2[k][j];
        }
    }
}

vector<vector<int>> matMul(vector<vector<int>> &mat1, vector<vector<int>> &mat2)
{
    int size = mat1.size();

    vector<vector<int>> result(size, vector<int>(size));
    vector<thread> threadsPool;

    fill(result.begin(), result.end(), vector<int>(size, 0));

    // Multiplying matrix aMat and secondMatrix and storing in array mult.
    for (int i = 0; i < size; i++)
    {
        // threadsPool.push_back(thread(thread_dot, ref(result), ref(mat1), ref(mat2), i));
        threadsPool.push_back(thread(thread_dot, ref(result), ref(mat1), ref(mat2), i));
    }

    for (thread &th : threadsPool)
    {
        if (th.joinable())
            th.join();
    }

    return result;
}

void fillMat(vector<vector<int>> &mat)
{
    int size = mat.size();

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            mat[i][j] = rand() % 20 + 1;
        }
    }
}

int main(int argc, char *argv[])
{
    // TODO get shape from params
    int size = atoi(argv[1]);
    vector<vector<int>> mat1(size, vector<int>(size));
    vector<vector<int>> mat2(size, vector<int>(size));

    // Fill matrix with random numbers
    fillMat(mat1);
    fillMat(mat2);

    // display(mat1);
    // display(mat2);

    auto start = std::chrono::high_resolution_clock::now();

    vector<vector<int>> result = matMul(mat1, mat2);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = finish - start;

    // display(result);

    // cout.precision(8);
    cout << size << " , " << elapsed.count() << endl;

    return 0;
}