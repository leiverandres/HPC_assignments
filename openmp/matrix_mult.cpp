#include <iostream>
#include <cstdlib>
#include <time.h>
#include <omp.h>
#define N 723

using namespace std;

void multSeq(int a[N][N], int b[N][N], int c[N][N]) {
    int sum;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum = 0;
            for (int k = 0; k < N; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

void multOmp(int a[N][N], int b[N][N], int c[N][N]) {
    int sum = 0, i, j, k, n = N;
    #pragma omp parallel shared(a, b, c, n), private(i, j, k, sum)
    {
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                sum = 0;
                for (k = 0; k < n; k++) {
                    sum += a[i][k] * b[k][j];
                }
                c[i][j] = sum;
            }
        }
    }
}

void showMat(int x[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << x[i][j] << " ";
        }
        cout << endl;
    }
}

void initMat(int x[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x[i][j] = rand() % 50;
        }
    }
}

bool checkResults(int s_c[N][N], int o_c[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (s_c[i][j] != o_c[i][j])
                return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    int a[N][N], b[N][N], s_c[N][N], o_c[N][N];
    double start_seq, start_omp, end_seq, end_omp;
    initMat(a);
    initMat(b);

    start_seq = omp_get_wtime( );
    multSeq(a, b, s_c);
    end_seq = omp_get_wtime( );

    start_omp = omp_get_wtime( );
    multOmp(a, b, o_c);
    end_omp = omp_get_wtime( );
    double seq_time = end_seq - start_seq;
    double omp_time = end_omp - start_omp;

    if (checkResults(s_c, o_c)) {
        cout << "Correct multiplication" << endl;
        cout << "Time for sequential: " << seq_time << " sec" << endl;
        cout << "Time for omp code: " << omp_time << " sec" << endl;
    } else {
        cout << "Incorrect multiplication" << endl;
    }
    return 0;
}
