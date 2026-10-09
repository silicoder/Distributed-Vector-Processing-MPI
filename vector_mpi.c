#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_VECTOR_SIZE 1000000LL
#define MAX_VECTOR_SIZE INT_MAX
#define DEFAULT_RESULTS_FILE "results.csv"

static int parse_vector_size(const char *text, long long *value) {
    char *end = NULL;
    errno = 0;
    long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > MAX_VECTOR_SIZE) {
        return 0;
    }
    *value = parsed;
    return 1;
}

static int append_result(const char *path, long long n, int processes,
                         double seconds, long long checksum) {
    FILE *file = fopen(path, "a+");
    if (file == NULL) {
        fprintf(stderr, "Warning: could not open results file '%s': %s\n", path, strerror(errno));
        return 0;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "Warning: could not seek results file '%s'.\n", path);
        fclose(file);
        return 0;
    }
    long position = ftell(file);
    if (position == 0) {
        fprintf(file, "mode,vector_size,processes,seconds,checksum\n");
    }
    fprintf(file, "MPI,%lld,%d,%.9f,%lld\n", n, processes, seconds, checksum);
    if (fclose(file) != 0) {
        fprintf(stderr, "Warning: could not finish writing '%s'.\n", path);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int rank = 0;
    int processes = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    long long n = DEFAULT_VECTOR_SIZE;
    const char *results_path = DEFAULT_RESULTS_FILE;
    int valid_args = (argc <= 3);
    if (argc >= 2) {
        valid_args = valid_args && parse_vector_size(argv[1], &n);
    }
    if (argc >= 3) {
        results_path = argv[2];
    }
    if (!valid_args) {
        if (rank == 0) {
            fprintf(stderr,
                    "Usage: mpirun -np <processes> ./vector_mpi [vector_size] [results.csv]\n"
                    "vector_size must be between 1 and %d.\n", MAX_VECTOR_SIZE);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int *counts = (int *)malloc((size_t)processes * sizeof(int));
    int *displacements = (int *)malloc((size_t)processes * sizeof(int));
    int setup_ok = (counts != NULL && displacements != NULL);
    long long *full_vector = NULL;
    long long *local_vector = NULL;
    long long *partial_sums = NULL;

    long long base_count = n / processes;
    long long remainder = n % processes;
    long long offset = 0;
    if (setup_ok) {
        for (int p = 0; p < processes; ++p) {
            long long chunk = base_count + (p < remainder ? 1 : 0);
            counts[p] = (int)chunk;
            displacements[p] = (int)offset;
            offset += chunk;
        }
    }

    int local_count = setup_ok ? counts[rank] : 0;
    size_t local_capacity = (size_t)(local_count > 0 ? local_count : 1);
    local_vector = (long long *)malloc(local_capacity * sizeof(long long));
    if (local_vector == NULL) {
        setup_ok = 0;
    }

    if (rank == 0 && setup_ok) {
        if ((unsigned long long)n > (unsigned long long)(SIZE_MAX / sizeof(long long))) {
            setup_ok = 0;
        } else {
            full_vector = (long long *)malloc((size_t)n * sizeof(long long));
            if (full_vector == NULL) {
                setup_ok = 0;
            } else {
                for (long long i = 0; i < n; ++i) {
                    full_vector[i] = i;
                }
            }
        }
        partial_sums = (long long *)malloc((size_t)processes * sizeof(long long));
        if (partial_sums == NULL) {
            setup_ok = 0;
        }
    }

    int every_process_ready = 0;
    MPI_Allreduce(&setup_ok, &every_process_ready, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!every_process_ready) {
        if (rank == 0) {
            fprintf(stderr, "Error: memory allocation failed. Try a smaller vector size.\n");
        }
        free(counts);
        free(displacements);
        free(full_vector);
        free(local_vector);
        free(partial_sums);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    MPI_Scatterv(full_vector, counts, displacements, MPI_LONG_LONG_INT,
                 local_vector, local_count, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < local_count; ++i) {
        local_sum += local_vector[i];
    }

    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT,
               partial_sums, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start_time;

    if (rank == 0) {
        long long global_sum = 0;
        printf("Distributed Vector Processing using MPI\n");
        printf("Vector size: %lld | MPI processes: %d\n", n, processes);
        printf("------------------------------------------------------------\n");
        printf("Rank | Elements | Index range       | Local sum\n");
        printf("------------------------------------------------------------\n");
        for (int p = 0; p < processes; ++p) {
            int count = counts[p];
            int first = displacements[p];
            int last = (count > 0) ? first + count - 1 : first - 1;
            global_sum += partial_sums[p];
            if (count > 0) {
                printf("%4d | %8d | %7d - %-7d | %lld\n",
                       p, count, first, last, partial_sums[p]);
            } else {
                printf("%4d | %8d | (no elements)      | %lld\n", p, count, partial_sums[p]);
            }
        }
        long long expected = (n * (n - 1)) / 2;
        printf("------------------------------------------------------------\n");
        printf("Global sum: %lld\n", global_sum);
        printf("Expected sum: %lld\n", expected);
        printf("Correctness: %s\n", global_sum == expected ? "PASS" : "FAIL");
        printf("MPI elapsed time (scatter + local sum + gather): %.9f seconds\n", elapsed);
        printf("Result row: mode=MPI, vector_size=%lld, processes=%d, seconds=%.9f, checksum=%lld\n",
               n, processes, elapsed, global_sum);
        append_result(results_path, n, processes, elapsed, global_sum);
        if (global_sum != expected) {
            fprintf(stderr, "Error: computed sum did not match the expected result.\n");
        }
    } else {
        char host[MPI_MAX_PROCESSOR_NAME];
        int host_length = 0;
        MPI_Get_processor_name(host, &host_length);
        printf("Rank %d running on %s: processed %d elements; local sum = %lld\n",
               rank, host, local_count, local_sum);
    }

    free(counts);
    free(displacements);
    free(full_vector);
    free(local_vector);
    free(partial_sums);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
