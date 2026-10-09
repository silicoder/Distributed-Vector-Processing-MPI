/*
 * Sequential baseline for Distributed Vector Processing using MPI.
 *
 * The vector uses the same data as vector_mpi.c: vector[i] = i.
 * The program records its measured execution time in results.csv so that
 * plot_results.py can compare sequential and MPI execution.
 *
 * Usage:
 *     ./vector_sequential [vector_size] [results.csv]
 */

#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static double seconds_between(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0;
}

static int append_result(const char *path, long long n, double seconds, long long checksum) {
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
    fprintf(file, "Sequential,%lld,1,%.9f,%lld\n", n, seconds, checksum);
    if (fclose(file) != 0) {
        fprintf(stderr, "Warning: could not finish writing '%s'.\n", path);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    long long n = DEFAULT_VECTOR_SIZE;
    const char *results_path = DEFAULT_RESULTS_FILE;
    if (argc > 3 || (argc >= 2 && !parse_vector_size(argv[1], &n))) {
        fprintf(stderr,
                "Usage: ./vector_sequential [vector_size] [results.csv]\n"
                "vector_size must be between 1 and %d.\n", MAX_VECTOR_SIZE);
        return EXIT_FAILURE;
    }
    if (argc >= 3) {
        results_path = argv[2];
    }

    if ((unsigned long long)n > (unsigned long long)(SIZE_MAX / sizeof(long long))) {
        fprintf(stderr, "Error: requested vector is too large for this platform.\n");
        return EXIT_FAILURE;
    }
    long long *vector = (long long *)malloc((size_t)n * sizeof(long long));
    if (vector == NULL) {
        fprintf(stderr, "Error: could not allocate the vector. Try a smaller size.\n");
        return EXIT_FAILURE;
    }
    for (long long i = 0; i < n; ++i) {
        vector[i] = i;
    }

    struct timespec start, end;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        perror("clock_gettime");
        free(vector);
        return EXIT_FAILURE;
    }
    long long sum = 0;
    for (long long i = 0; i < n; ++i) {
        sum += vector[i];
    }
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        perror("clock_gettime");
        free(vector);
        return EXIT_FAILURE;
    }

    double elapsed = seconds_between(start, end);
    long long expected = (n * (n - 1)) / 2;
    printf("Sequential Vector Processing\n");
    printf("Vector size: %lld\n", n);
    printf("Global sum: %lld\n", sum);
    printf("Expected sum: %lld\n", expected);
    printf("Correctness: %s\n", sum == expected ? "PASS" : "FAIL");
    printf("Sequential elapsed time (sum loop): %.9f seconds\n", elapsed);
    printf("Result row: mode=Sequential, vector_size=%lld, processes=1, seconds=%.9f, checksum=%lld\n",
           n, elapsed, sum);
    append_result(results_path, n, elapsed, sum);

    free(vector);
    return (sum == expected) ? EXIT_SUCCESS : EXIT_FAILURE;
}
