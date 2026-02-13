# Alpha Formula Comparison: Sigmoid vs Linear

This document summarizes the behavioral differences between the new sigmoid-based alpha formula and the legacy linear formula.

## Comparison Table

| Scenario | N | C_g | Linear | Sigmoid | Difference |
|----------|---|-----|--------|---------|------------|
| New user, low conf | 0 | 0.5 | 1.000 | 0.269 | 0.731 |
| New user, med conf | 0 | 0.7 | 1.000 | 0.731 | 0.269 |
| New user, high conf | 0 | 0.9 | 1.000 | 0.953 | 0.047 |
| Medium feedback, low conf | 50 | 0.5 | 0.550 | 0.134 | 0.416 |
| Medium feedback, med conf | 50 | 0.7 | 0.610 | 0.366 | 0.244 |
| Medium feedback, high conf | 50 | 0.9 | 0.670 | 0.476 | 0.194 |
| High feedback, low conf | 100 | 0.5 | 0.450 | 0.090 | 0.360 |
| High feedback, med conf | 100 | 0.7 | 0.510 | 0.244 | 0.266 |
| High feedback, high conf | 100 | 0.9 | 0.570 | 0.318 | 0.252 |

## Key Observations

### 1. New User Behavior (N=0)

**Linear Formula:**
- Always returns α = 1.0 for N < 20 (hardcoded threshold)
- Completely ignores confidence for new users
- Assumes global head is always trustworthy for new users

**Sigmoid Formula:**
- Responds to confidence even for new users
- Low confidence (C_g=0.5): α = 0.269 (favors user head)
- High confidence (C_g=0.9): α = 0.953 (favors global head)
- More nuanced decision-making from the start

**Impact:** Sigmoid is 2.5-55x more conservative for new users with low confidence.

### 2. Medium Feedback (N=50)

**Linear Formula:**
- α ranges from 0.55 to 0.67 across confidence levels
- Relatively narrow range (0.18 spread)
- Still favors global head moderately

**Sigmoid Formula:**
- α ranges from 0.134 to 0.476 across confidence levels
- Much wider range (0.45 spread)
- **2.5x more sensitive to confidence changes**
- Favors user head more aggressively

**Impact:** Sigmoid provides stronger personalization at medium feedback levels.

### 3. High Feedback (N=100)

**Linear Formula:**
- α ranges from 0.45 to 0.57
- Still gives significant weight to global head
- Minimum alpha is 0.3 (clamped)

**Sigmoid Formula:**
- α ranges from 0.090 to 0.318
- Strongly favors user head
- No artificial clamping needed

**Impact:** Sigmoid trusts personalized model much more with sufficient feedback.

### 4. Discontinuity at N=20

**Linear Formula:**
- Has discontinuous jump at N=20
- α drops from 1.0 to ~0.71 instantly
- Creates abrupt behavior change

**Sigmoid Formula:**
- Smooth exponential decay throughout
- No discontinuities or jumps
- Gradual transition from global to user head

**Impact:** Sigmoid provides smoother user experience.

### 5. Multiplicative vs Additive Logic

**Linear Formula (Additive):**
```
α = 0.5 + 0.3·C_g - 0.2·min(N/100, 1.0)
```
- Components can partially cancel each other
- Less conservative with low confidence

**Sigmoid Formula (Multiplicative):**
```
α = alpha_data × alpha_conf
```
- Both components must be high for high α
- More conservative (AND logic)
- Low confidence strongly reduces α

**Example (N=50, C_g=0.3):**
- Linear: α = 0.49 (moderate)
- Sigmoid: α = 0.025 (very low)
- **Sigmoid is 20x more conservative**

### 6. Confidence Sensitivity

**Linear:** 0.18 range across confidence values (at N=50)
**Sigmoid:** 0.45 range across confidence values (at N=50)

**Sigmoid is 2.5x more sensitive to confidence changes.**

This means sigmoid responds more dramatically to model uncertainty, providing better risk management.

### 7. Extreme Scenarios

| Scenario | Linear | Sigmoid | Ratio |
|----------|--------|---------|-------|
| N=0, C_g=0.2 (new user, very low conf) | 1.000 | 0.018 | 55.6x |
| N=0, C_g=0.95 (new user, very high conf) | 1.000 | 0.971 | 1.03x |
| N=200, C_g=0.2 (lots of feedback, very low conf) | 0.360 | 0.004 | >100x |
| N=200, C_g=0.95 (lots of feedback, very high conf) | 0.585 | 0.194 | 3.0x |

**Key Insight:** Sigmoid is dramatically more conservative with low confidence, regardless of feedback count.

## Summary

### Sigmoid Formula Advantages

1. **Confidence-Aware from Start:** Responds to confidence even for new users
2. **Smoother Transitions:** No discontinuities or hardcoded thresholds
3. **More Aggressive Personalization:** Favors user head more strongly with feedback
4. **Better Risk Management:** Multiplicative logic is more conservative with uncertainty
5. **Higher Sensitivity:** 2.5x more responsive to confidence changes
6. **Natural Bounds:** No artificial clamping needed

### When Sigmoid Differs Most

- **New users with low confidence:** 55x more conservative
- **Medium feedback with low confidence:** 20x more conservative
- **Any scenario with low confidence:** Dramatically favors user head

### Migration Considerations

- Sigmoid will shift predictions toward user head in most scenarios
- Most pronounced impact on users with 20-100 feedback samples
- High-confidence predictions remain similar
- Low-confidence predictions will rely much more on user head

## Testing

Run comparison tests:
```bash
cd api
python -m pytest tests/test_alpha_formula_comparison.py -v -s
```

The `-s` flag shows the detailed comparison tables during test execution.
