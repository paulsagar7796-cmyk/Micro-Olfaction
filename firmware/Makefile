# System 2+ firmware: the library, the scenario harness (harness.c) and the cycle benchmarks (bench.c,
# with DA-ELM), for the host and for a Cortex-M4F (STM32F405 memory map, run under QEMU -M
# netduinoplus2). check_firmware.py and cm4_profile.py drive these builds.
#
# VARIANT=base (default): the portable reference, s2p.c and elm.c.
# VARIANT=opt: the optimised kernels, s2p_opt.c and elm_opt.c, bit-identical to the reference; their
# builds carry the suffix _opt.
#
# No floating-point contraction on either target: fused multiply-adds would change the rounding, and
# the results must match device_reference.py bit for bit.
# If the host link fails with "tapi error ... unknown architecture" (a Command Line Tools linker older
# than the default macOS SDK), build against an older SDK, e.g.
#   SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk make
# (the Python drivers do this automatically). ARM_TOOLS / QEMU_INCLUDE: the Arm GCC bin directory and
# QEMU's include directory (qemu-plugin.h), if not on the default paths.

VARIANT ?= base
ifeq ($(VARIANT),opt)
S2P_SRC = s2p_opt.c
ELM_SRC = elm_opt.c
SUFFIX = _opt
ELM_CHECK_FLAGS = -DELM_SCORES
else
S2P_SRC = s2p.c
ELM_SRC = elm.c
SUFFIX =
ELM_CHECK_FLAGS =
endif

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
HEADERS = s2p.h io.h scenario.h elm.h fastmath.h

host: build/host_harness$(SUFFIX)
cm4: build/cm4_harness$(SUFFIX).elf build/cm4_bench$(SUFFIX).elf build/s2p$(SUFFIX)_cm4_50.o

build/host_harness$(SUFFIX): harness.c scenario.c io_host.c $(S2P_SRC) $(HEADERS)
	mkdir -p build
	$(CC) $(CFLAGS) -DS2P_MAX_LABELLED=$(MAX_LABELLED) -o $@ harness.c scenario.c io_host.c $(S2P_SRC)

build/elm_check$(SUFFIX): elm_check.c $(ELM_SRC) $(S2P_SRC) build/elm_weights.c $(HEADERS)
	$(CC) $(CFLAGS) $(ELM_CHECK_FLAGS) -I. -o $@ elm_check.c $(ELM_SRC) $(S2P_SRC) build/elm_weights.c

# Objects for the chip, compiled in build/cm4 (so that the -fstack-usage reports land there)
build/cm4/%.o: %.c $(HEADERS)
	mkdir -p $(dir $@)
	cd $(dir $@) && $(ARM_CC) $(ARM_FLAGS) $(ARM_INC) -DS2P_MAX_LABELLED=$(MAX_LABELLED) -c "$(CURDIR)/$<"

CM4_LIB = build/cm4/$(S2P_SRC:.c=.o)
CM4_ELM = build/cm4/$(ELM_SRC:.c=.o)

# The library first: its code is one contiguous block, which cm4_profile.py counts
build/cm4_harness$(SUFFIX).elf: $(CM4_LIB) build/cm4/harness.o build/cm4/scenario.o build/cm4/cm4/io_semihost.o \
                                build/cm4/cm4/startup.o cm4/stm32f405.ld
	$(ARM_CC) $(ARM_FLAGS) $(ARM_LINK) -Wl,-Map=build/cm4_harness$(SUFFIX).map -o $@ $(filter %.o,$^)

build/cm4/elm_weights.o: build/elm_weights.c
	mkdir -p build/cm4
	$(ARM_CC) $(ARM_FLAGS) -c -o $@ $<

build/cm4_bench$(SUFFIX).elf: $(CM4_LIB) $(CM4_ELM) build/cm4/bench.o build/cm4/scenario.o \
                              build/cm4/cm4/io_semihost.o build/cm4/cm4/startup.o build/cm4/elm_weights.o cm4/stm32f405.ld
	$(ARM_CC) $(ARM_FLAGS) $(ARM_LINK) -Wl,-Map=build/cm4_bench$(SUFFIX).map -o $@ $(filter %.o,$^)

# The library alone at the paper's configuration (50 labelled samples), for its code size and stack
build/s2p$(SUFFIX)_cm4_50.o: $(S2P_SRC) $(HEADERS)
	mkdir -p build
	$(ARM_CC) $(ARM_FLAGS) $(ARM_INC) -DS2P_MAX_LABELLED=50 -c -o $@ $(S2P_SRC)

# The profiling plugin for QEMU (host)
build/libcm4prof.dylib: cm4/profiler/cm4prof.c cm4/profiler/glib.h
	mkdir -p build
	$(CC) -shared -fPIC -O2 -Wall -Wextra -Icm4/profiler -I$(QEMU_INCLUDE) -undefined dynamic_lookup -o $@ $<

clean:
	rm -rf build

.PHONY: host cm4 clean
