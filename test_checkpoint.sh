#!/bin/bash
# Test script for FastTree checkpoint/restart functionality.
# Verifies bit-identical output for single-threaded runs across
# different models and settings, plus multi-threaded completion.
#
# Usage: ./test_checkpoint.sh [protein_alignment] [nucleotide_alignment]
# Defaults: BigCOGs/COG1011.500.p and 16S500/16S.1.p

set -euo pipefail

AA_ALN="${1:-BigCOGs/COG1011.500.p}"
NT_ALN="${2:-16S500/16S.1.p}"
FASTTREE="./FastTree"
TMPDIR=$(mktemp -d)
PASS=0
FAIL=0

cleanup() {
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

fail_test() {
    echo "FAIL: $1"
    FAIL=$((FAIL + 1))
}

pass_test() {
    echo "PASS: $1"
    PASS=$((PASS + 1))
}

echo "=== Compiling FastTree ==="
gcc -DOPENMP -O3 -fopenmp -fopenmp-simd -funsafe-math-optimizations -march=native \
    -o "$FASTTREE" FastTree.c -lm
echo "Compiled OK"
echo ""

# run_checkpoint_test NAME ARGS... ALN
#   1. Full reference run (no checkpoint)
#   2. Full run with checkpoint (should match reference)
#   3. Early restart (kill after 15% of ref time)
#   4. Late restart (from full run's last checkpoint)
run_checkpoint_test() {
    local name="$1"
    shift
    local aln="${!#}"             # last argument is alignment
    local ft_args="${@:1:$#-1}"  # everything except last arg

    echo "--- $name ---"

    # Reference run
    OMP_NUM_THREADS=1 "$FASTTREE" $ft_args "$aln" \
        > "$TMPDIR/${name}_ref.tree" 2>"$TMPDIR/${name}_ref.stderr"
    local ref_time
    ref_time=$(grep "Total time:" "$TMPDIR/${name}_ref.stderr" | sed 's/Total time: \([0-9.]*\).*/\1/')
    echo "  Reference: ${ref_time}s"

    # Full checkpoint run
    OMP_NUM_THREADS=1 "$FASTTREE" $ft_args -checkpoint "$TMPDIR/${name}_ckpt.bin" "$aln" \
        > "$TMPDIR/${name}_ckpt.tree" 2>"$TMPDIR/${name}_ckpt.stderr"
    if diff -q "$TMPDIR/${name}_ref.tree" "$TMPDIR/${name}_ckpt.tree" > /dev/null 2>&1; then
        pass_test "$name: checkpoint run matches reference"
    else
        fail_test "$name: checkpoint run differs from reference"
    fi

    # Early restart (kill after 15% of ref time, min 2s)
    local early_timeout
    early_timeout=$(python3 -c "t=int(float('$ref_time')*0.15); print(max(t,2))")
    timeout "$early_timeout" bash -c \
        "OMP_NUM_THREADS=1 $FASTTREE $ft_args -checkpoint $TMPDIR/${name}_early.bin $aln > /dev/null 2>$TMPDIR/${name}_early.stderr" || true
    if [ -f "$TMPDIR/${name}_early.bin" ]; then
        local phase
        phase=$(grep "Checkpoint saved" "$TMPDIR/${name}_early.stderr" | tail -1 | sed 's/.*phase \([0-9]*\).*/\1/')
        local round
        round=$(grep "Checkpoint saved" "$TMPDIR/${name}_early.stderr" | tail -1 | sed 's/.*round \([0-9]*\).*/\1/')
        echo "  Early checkpoint: phase $phase round $round (timeout ${early_timeout}s)"
        OMP_NUM_THREADS=1 "$FASTTREE" $ft_args -restart "$TMPDIR/${name}_early.bin" "$aln" \
            > "$TMPDIR/${name}_restart_early.tree" 2>"$TMPDIR/${name}_restart_early.stderr"
        if diff -q "$TMPDIR/${name}_ref.tree" "$TMPDIR/${name}_restart_early.tree" > /dev/null 2>&1; then
            pass_test "$name: early restart (phase $phase) matches reference"
        else
            fail_test "$name: early restart (phase $phase) differs from reference"
        fi
    else
        echo "  WARNING: no early checkpoint created (timeout ${early_timeout}s too short for NJ)"
    fi

    # Late restart (from full run's last checkpoint)
    local late_phase
    late_phase=$(grep "Checkpoint saved" "$TMPDIR/${name}_ckpt.stderr" | tail -1 | sed 's/.*phase \([0-9]*\).*/\1/')
    local late_round
    late_round=$(grep "Checkpoint saved" "$TMPDIR/${name}_ckpt.stderr" | tail -1 | sed 's/.*round \([0-9]*\).*/\1/')
    echo "  Late checkpoint: phase $late_phase round $late_round"
    OMP_NUM_THREADS=1 "$FASTTREE" $ft_args -restart "$TMPDIR/${name}_ckpt.bin" "$aln" \
        > "$TMPDIR/${name}_restart_late.tree" 2>"$TMPDIR/${name}_restart_late.stderr"
    if diff -q "$TMPDIR/${name}_ref.tree" "$TMPDIR/${name}_restart_late.tree" > /dev/null 2>&1; then
        pass_test "$name: late restart (phase $late_phase) matches reference"
    else
        fail_test "$name: late restart (phase $late_phase) differs from reference"
    fi
    echo ""
}

echo "=== Single-threaded tests (OMP_NUM_THREADS=1) ==="
echo ""

# Protein models
run_checkpoint_test "protein_jtt"       "$AA_ALN"
run_checkpoint_test "protein_wag" -wag  "$AA_ALN"
run_checkpoint_test "protein_lg"  -lg   "$AA_ALN"

# Protein with different rate models
run_checkpoint_test "protein_nocat" -nocat "$AA_ALN"
run_checkpoint_test "protein_gamma" -gamma "$AA_ALN"

# Protein without ML
run_checkpoint_test "protein_noml" -noml "$AA_ALN"

# Nucleotide models
run_checkpoint_test "nucleotide_jc"  -nt       "$NT_ALN"
run_checkpoint_test "nucleotide_gtr" -nt -gtr  "$NT_ALN"

# Nucleotide with gamma
run_checkpoint_test "nucleotide_gtr_gamma" -nt -gtr -gamma "$NT_ALN"

echo "=== Multi-threaded test ==="
echo ""

# Multi-threaded checkpoint + restart (verify completion, not bit-identity)
echo "--- Multi-threaded protein ---"
"$FASTTREE" -checkpoint "$TMPDIR/mt_ckpt.bin" "$AA_ALN" \
    > "$TMPDIR/mt_full.tree" 2>"$TMPDIR/mt_full.stderr"
MT_TIME=$(grep "Total time:" "$TMPDIR/mt_full.stderr" | sed 's/Total time: \([0-9.]*\).*/\1/')
echo "  Full run: ${MT_TIME}s"

"$FASTTREE" -restart "$TMPDIR/mt_ckpt.bin" "$AA_ALN" \
    > "$TMPDIR/mt_restart.tree" 2>"$TMPDIR/mt_restart.stderr"
MT_RESTART_SIZE=$(wc -c < "$TMPDIR/mt_restart.tree")
if [ "$MT_RESTART_SIZE" -gt 0 ] && grep -q "Total time:" "$TMPDIR/mt_restart.stderr"; then
    pass_test "Multi-threaded restart completed ($MT_RESTART_SIZE bytes)"
else
    fail_test "Multi-threaded restart failed or produced empty output"
fi

echo "--- Multi-threaded nucleotide GTR ---"
"$FASTTREE" -nt -gtr -checkpoint "$TMPDIR/mt_nt_ckpt.bin" "$NT_ALN" \
    > "$TMPDIR/mt_nt_full.tree" 2>"$TMPDIR/mt_nt_full.stderr"
MT_NT_TIME=$(grep "Total time:" "$TMPDIR/mt_nt_full.stderr" | sed 's/Total time: \([0-9.]*\).*/\1/')
echo "  Full run: ${MT_NT_TIME}s"

"$FASTTREE" -nt -gtr -restart "$TMPDIR/mt_nt_ckpt.bin" "$NT_ALN" \
    > "$TMPDIR/mt_nt_restart.tree" 2>"$TMPDIR/mt_nt_restart.stderr"
MT_NT_SIZE=$(wc -c < "$TMPDIR/mt_nt_restart.tree")
if [ "$MT_NT_SIZE" -gt 0 ] && grep -q "Total time:" "$TMPDIR/mt_nt_restart.stderr"; then
    pass_test "Multi-threaded nucleotide GTR restart completed ($MT_NT_SIZE bytes)"
else
    fail_test "Multi-threaded nucleotide GTR restart failed"
fi

echo ""
echo "=== Results ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "SOME TESTS FAILED"
    exit 1
else
    echo "ALL TESTS PASSED"
    exit 0
fi
