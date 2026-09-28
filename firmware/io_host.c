/* Scenario I/O on the host: stdio. */
#include <stdio.h>
#include <stdlib.h>

#include "io.h"

static FILE *in, *out;

void io_fail(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(2);
}

void io_open(const char *in_path, const char *out_path) {
    if (!(in = fopen(in_path, "rb")) || !(out = fopen(out_path, "wb"))) io_fail("cannot open scenario or output");
}

void io_read(void *p, size_t n) {
    if (n && fread(p, 1, n, in) != n) io_fail("scenario truncated");
}

void io_write(const void *p, size_t n) {
    if (n && fwrite(p, 1, n, out) != n) io_fail("cannot write output");
}

void io_close(void) {
    fclose(in);
    fclose(out);
}
