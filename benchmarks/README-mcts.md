# MCTS CPU benchmarks

Build and run from the repository root:

```sh
lake build mctx_bench
lake env ./.lake/build/bin/mctx_bench
```

The executable checks correctness before reporting CSV timings. It contains two
separate comparisons:

- **Allocation:** the former linear scan for the next unused node versus the
  maintained allocation counter. Each sample fills an arena and checks the sum
  of allocated indices. This is an isolated allocator benchmark.
- **Gumbel search:** the former root selector that constructs the entire
  sequential-halving table on every simulation versus a cached schedule. Both
  paths use the same current search, allocation counter, deterministic four-action
  model, and depth limit of 12. Every node's visits and values, child indices and
  visits, and root Q values must agree exactly. Root visits must account for the
  complete simulation budget.

The search comparison measures complete synthetic CPU searches, including
selection, model evaluation and backup. It isolates schedule caching; it does
not measure a combined speedup from caching and allocation. Neither comparison
measures GPU inference, a real application, or planning quality.

## Recorded result

[Raw CSV](results/mcts_m3_max_2026-09-08.csv), recorded on 2026-09-08 with an Apple
M3 Max, arm64 macOS 26.3.1, and Lean 4.29.0 (release commit
`98dc76e3c0a9b856c9b98726b713fb04fab16740`). The code was built natively with the
commands above and the default Lake release settings.

| Case | Size | Baseline (ms) | Optimized (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Arena allocation | 128 | 0.018529 | 0.002608 | 7.10x |
| Arena allocation | 512 | 0.211188 | 0.009513 | 22.20x |
| Arena allocation | 2,048 | 3.035258 | 0.038829 | 78.17x |
| Arena allocation | 8,192 | 47.190492 | 0.143863 | 328.02x |
| Gumbel search | 128 | 11.004208 | 6.853400 | 1.61x |
| Gumbel search | 512 | 98.645975 | 35.822500 | 2.75x |
| Gumbel search | 1,024 | 316.086542 | 73.668208 | 4.29x |

Times are arithmetic means of 10 allocator samples or 5 search samples. Samples
alternate the listed size and size plus one to keep repeated work observable;
both search budgets pass the equivalence check. Baseline samples run first.
These are observations from one run, without confidence intervals or a CI timing
threshold. Rerun on the target machine before using the numbers for capacity
planning.
