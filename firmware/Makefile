# Host build of the System 2+ library and its test harness (check_firmware.py runs this).
# No floating-point contraction: fused multiply-adds would change the rounding, and the results must
# match device_reference.py bit for bit.
# If linking fails with "tapi error ... unknown architecture" (a Command Line Tools linker older than
# the default macOS SDK), build against an older SDK, e.g.
#   SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk make
# (check_firmware.py does this automatically).
CC ?= cc
CFLAGS = -std=c99 -O2 -ffp-contract=off -fno-fast-math -fno-math-errno -Wall -Wextra -Wdouble-promotion -Werror
MAX_LABELLED ?= 96

build/host_harness: host_harness.c s2p.c s2p.h
	mkdir -p build
	$(CC) $(CFLAGS) -DS2P_MAX_LABELLED=$(MAX_LABELLED) -o $@ host_harness.c s2p.c

clean:
	rm -rf build

.PHONY: clean
