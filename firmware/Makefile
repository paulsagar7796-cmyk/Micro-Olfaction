# System 2+ firmware: the library (s2p.c), the scenario harness (harness.c) and the cycle benchmarks
# (bench.c, with DA-ELM in elm.c), built for the host and for a Cortex-M4F (STM32F405 memory map, run
# under QEMU -M netduinoplus2). check_firmware.py and cm4_profile.py drive these builds.
#
# No floating-point contraction on either target: fused multiply-adds would change the rounding, and
# the results must match device_reference.py bit for bit.
# If the host link fails with "tapi error ... unknown architecture" (a Command Line Tools linker older
# than the default macOS SDK), build against an older SDK, e.g.
#   SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk make
# (the Python drivers do this automatically). ARM_TOOLS / QEMU_INCLUDE: the Arm GCC bin directory and
# QEMU's include directory (qemu-plugin.h), if not on the default paths.

CC ?= cc
CFLAGS = -std=c99 -O2 -ffp-contract=off -fno-fast-math -fno-math-errno -Wall -Wextra -Wdouble-promotion -Werror
MAX_LABELLED ?= 96

ARM_TOOLS ?=
ARM_CC = $(if $(ARM_TOOLS),$(ARM_TOOLS)/)arm-none-eabi-gcc
ARM_FLAGS = -std=c99 -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -ffp-contract=off \
            -fno-math-errno -ffreestanding -fno-tree-loop-distribute-patterns -fno-common \
            -ffunction-sections -fdata-sections -fstack-usage -g \
            -Wall -Wextra -Wdouble-promotion -Werror
ARM_INC = -I"$(CURDIR)/cm4/include" -I"$(CURDIR)"
ARM_LINK = -nostdlib -T cm4/stm32f405.ld -Wl,--gc-sections
QEMU_INCLUDE ?= /usr/local/include

host: build/host_harness
cm4: build/cm4_harness.elf build/cm4_bench.elf build/s2p_cm4_50.o

build/host_harness: harness.c scenario.c io_host.c s2p.c s2p.h io.h scenario.h
	mkdir -p build
	$(CC) $(CFLAGS) -DS2P_MAX_LABELLED=$(MAX_LABELLED) -o $@ harness.c scenario.c io_host.c s2p.c

# Objects for the chip, compiled in build/cm4 (so that the -fstack-usage reports land there)
CM4_COMMON = s2p.c scenario.c cm4/io_semihost.c cm4/startup.c
build/cm4/%.o: %.c s2p.h io.h scenario.h elm.h
	mkdir -p $(dir $@)
	cd $(dir $@) && $(ARM_CC) $(ARM_FLAGS) $(ARM_INC) -DS2P_MAX_LABELLED=$(MAX_LABELLED) -c "$(CURDIR)/$<"

# The library first: its code is one contiguous block, which cm4_profile.py counts
build/cm4_harness.elf: build/cm4/s2p.o build/cm4/harness.o build/cm4/scenario.o build/cm4/cm4/io_semihost.o \
                       build/cm4/cm4/startup.o cm4/stm32f405.ld
	$(ARM_CC) $(ARM_FLAGS) $(ARM_LINK) -Wl,-Map=build/cm4_harness.map -o $@ $(filter %.o,$^)

build/cm4/elm_weights.o: build/elm_weights.c
	mkdir -p build/cm4
	$(ARM_CC) $(ARM_FLAGS) -c -o $@ $<

build/cm4_bench.elf: build/cm4/s2p.o build/cm4/elm.o build/cm4/bench.o build/cm4/scenario.o \
                     build/cm4/cm4/io_semihost.o build/cm4/cm4/startup.o build/cm4/elm_weights.o cm4/stm32f405.ld
	$(ARM_CC) $(ARM_FLAGS) $(ARM_LINK) -Wl,-Map=build/cm4_bench.map -o $@ $(filter %.o,$^)

# The library alone at the paper's configuration (50 labelled samples), for its code size
build/s2p_cm4_50.o: s2p.c s2p.h
	mkdir -p build
	$(ARM_CC) $(ARM_FLAGS) $(ARM_INC) -DS2P_MAX_LABELLED=50 -c -o $@ s2p.c

# The profiling plugin for QEMU (host)
build/libcm4prof.dylib: cm4/profiler/cm4prof.c cm4/profiler/glib.h
	mkdir -p build
	$(CC) -shared -fPIC -O2 -Wall -Wextra -Icm4/profiler -I$(QEMU_INCLUDE) -undefined dynamic_lookup -o $@ $<

clean:
	rm -rf build

.PHONY: host cm4 clean
