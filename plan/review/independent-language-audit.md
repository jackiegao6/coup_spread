# Independent Language Audit

## Scope

The complete manuscript was edited conservatively at sentence level. The edit
preserved notation, theorem statements, experimental values, the Introduction
structure, and the comparison/challenge format in Section 3. It did not add a
novelty claim or strengthen an empirical conclusion.

## Terminology decisions

- Activation and activated user are retained only for classical IC/LT
  diffusion.
- In CIM, a coupon is redeemed, and the redeemer becomes an adopter.
- Adoption spread denotes the expected number of distinct adopters.
- Total redemptions counts coupons; coupon usage rate divides this count by the
  coupon budget.
- Realization is used consistently for a fixed per-coupon action space.
- Time step replaces timestamp, and tracing backward replaces reversely
  sampling.

## Main corrections

- Replaced ambiguous pronouns and passive constructions in the Abstract,
  model, RR construction, and experimental discussion.
- Clarified that the main experimental comparison uses distinct seeds, while
  the capacity experiment relaxes this policy.
- Defined the relative-difference formula reported in the aggregate table.
- Distinguished the theorem-calibrated sample count from the fixed empirical
  schedule without changing either claim.
- Weakened the unsupported word unavoidable in the lower-bound discussion.
- Added the constant-time action-sampling implementation detail needed by the
  stated expected-time argument.
- Moved the two full-width result figures earlier in their source sections so
  that text fills the space beneath them.
- Moved the model figure's coverage box to the background layer so that it no
  longer hides the adopter nodes.

## Checks

- No remaining uses of timestamp, reversely sampling, activation spread, or
  adopts a coupon were found.
- The spell checker reports only mathematical commands, author names, dataset
  names, and domain-specific terms.
- The final PDF has no unresolved references, missing graphics, missing
  characters, or horizontal overfull boxes.
- Visual inspection covered the title page, model and hardness figures,
  approximation proof, aggregate table, all result figures, Conclusion, and
  References.

## Residual presentation risk

The final ACM build reports a 1.166 pt vertical overfull box while balancing
the last reference page. It is not visible in the rendered page and does not
clip text. Bibliography entries also trigger legacy BibTeX warnings for omitted
address fields; these do not affect citation resolution.
