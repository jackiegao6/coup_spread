# Project Overview

- Paper type: Computer-science conference research paper (ACM SIGMOD style).
- Topic: Coupon influence maximization under non-replicable, single-path diffusion.
- Main writing manuscript: `paper-zh.tex` (Chinese working draft, ACM LaTeX layout; compile with XeLaTeX).
- English reference: `paper-v2 copy.tex`; `paper-v2-bak.tex` was the identical source used to create the Chinese draft.
- Current objective: develop and revise the paper in Chinese first, preserving mathematical claims and traceable experiment evidence; return to English submission preparation afterward.
- Core claims affected: approximation and runtime guarantees, theorem-to-implementation boundaries, empirical quality and efficiency claims, and anonymous-review metadata.

## Chapter Structure

1. Introduction
2. Related Work
3. Model and Problem Definition
4. Theoretical Analysis
5. Algorithm: CIM-RIS
6. Other Variants
7. Experiments
8. Conclusion

## Chinese Working Draft

- `paper-zh.tex` translates the abstract, main text, proofs, pseudocode, captions, tables, and parameter-sweep appendix. Existing experimental image assets retain their English in-image labels. Bibliography records and official ACM CCS metadata are retained.
- Build from the repository root with `latexmk -xelatex -interaction=nonstopmode paper-zh.tex`. The source uses `ctex` with the Fandol font set, alongside the existing ACM dependencies.
- Source checks passed for 46 display equations (excluding translated natural-language text), label/reference/citation keys, table numbers, graphic paths, and environment nesting. No LaTeX compiler was found in the current local environment, so PDF compilation and visual layout remain unverified.
