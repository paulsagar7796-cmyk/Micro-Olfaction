/* Scenario I/O for the test harness: stdio on the host (io_host.c), Arm semihosting on the chip
 * (cm4/io_semihost.c). */
#ifndef S2P_IO_H
#define S2P_IO_H

#include <stddef.h>

void io_open(const char *in_path, const char *out_path);   /* exits on failure */
void io_read(void *p, size_t n);                            /* exits on a short read */
void io_write(const void *p, size_t n);
void io_close(void);
void io_fail(const char *msg);                              /* message, then exit status 2 */

#endif
