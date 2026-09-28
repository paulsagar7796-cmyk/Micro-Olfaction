/*
 * QEMU TCG plugin: the exact execution count of every instruction in [lo, hi), and how often each
 * block-ending instruction transferred control elsewhere (a taken branch), for the Cortex-M4 cycle
 * model in cm4_profile.py.
 *
 *   qemu-system-arm ... -plugin libcm4prof.dylib,lo=0x08000000,hi=0x08004000,out=profile.bin
 *
 * Output: uint64 triples (address, executions, taken) for each executed instruction in range.
 * Counting uses inline scoreboard increments (no callback per instruction); taken branches are found
 * with one callback per translation-block execution: a block that does not start at the previous
 * block's fall-through address was reached by a taken branch at the previous block's last instruction.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

typedef struct {
    uint64_t start, last, fall;
} tb_info;

static uint64_t lo, hi;
static size_t n_slots;                            /* one per halfword: Thumb instructions are 2 or 4 bytes */
static struct qemu_plugin_scoreboard *counts;     /* per vCPU: uint64_t[n_slots] */
static uint64_t *taken;                           /* single vCPU (system emulation of one core) */
static const tb_info *prev;
static char out_path[1024] = "profile.bin";

static void tb_exec(unsigned int vcpu, void *udata) {
    (void)vcpu;
    const tb_info *t = udata;
    if (prev && prev->last >= lo && prev->last < hi && t->start != prev->fall) taken[(prev->last - lo) / 2]++;
    prev = t;
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
    (void)id;
    size_t n = qemu_plugin_tb_n_insns(tb);
    if (n == 0) return;
    tb_info *t = malloc(sizeof *t);
    struct qemu_plugin_insn *last = qemu_plugin_tb_get_insn(tb, n - 1);
    t->start = qemu_plugin_tb_vaddr(tb);
    t->last = qemu_plugin_insn_vaddr(last);
    t->fall = t->last + qemu_plugin_insn_size(last);
    qemu_plugin_register_vcpu_tb_exec_cb(tb, tb_exec, QEMU_PLUGIN_CB_NO_REGS, t);
    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        uint64_t pc = qemu_plugin_insn_vaddr(insn);
        if (pc >= lo && pc < hi) {
            qemu_plugin_u64 slot = {counts, (size_t)((pc - lo) / 2) * sizeof(uint64_t)};
            qemu_plugin_register_vcpu_insn_exec_inline_per_vcpu(insn, QEMU_PLUGIN_INLINE_ADD_U64, slot, 1);
        }
    }
}

static void plugin_exit(qemu_plugin_id_t id, void *p) {
    (void)id; (void)p;
    const uint64_t *c = qemu_plugin_scoreboard_find(counts, 0);
    FILE *f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "cm4prof: cannot write %s\n", out_path); return; }
    for (size_t s = 0; s < n_slots; s++) {
        if (c[s] || taken[s]) {
            uint64_t rec[3] = {lo + 2 * s, c[s], taken[s]};
            fwrite(rec, sizeof(uint64_t), 3, f);
        }
    }
    fclose(f);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv) {
    (void)info;
    for (int i = 0; i < argc; i++) {
        if (!strncmp(argv[i], "lo=", 3)) lo = strtoull(argv[i] + 3, NULL, 0);
        else if (!strncmp(argv[i], "hi=", 3)) hi = strtoull(argv[i] + 3, NULL, 0);
        else if (!strncmp(argv[i], "out=", 4)) snprintf(out_path, sizeof out_path, "%s", argv[i] + 4);
        else { fprintf(stderr, "cm4prof: unknown option %s\n", argv[i]); return -1; }
    }
    if (hi <= lo) { fprintf(stderr, "cm4prof: need lo < hi\n"); return -1; }
    n_slots = (size_t)((hi - lo) / 2);
    counts = qemu_plugin_scoreboard_new(n_slots * sizeof(uint64_t));
    taken = calloc(n_slots, sizeof(uint64_t));
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    return 0;
}
