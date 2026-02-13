# Alpha Engine Migration Guide: Linear to Sigmoid Formula

## Overview

This guide provides step-by-step instructions for migrating from the legacy linear alpha formula to the new sigmoid-based formula in production. The sigmoid formula provides smoother transitions, better separation of concerns, and more intuitive tuning compared to the linear approach.

**Target Audience:** DevOps engineers, system operators, and production deployment teams

**Prerequisites:**
- Familiarity with the dual-head emotion recognition system
- Access to production configuration files
- Ability to restart services and monitor logs

---

## Quick Reference

### Enable Sigmoid Formula

Edit `api/app/services/feature_config.py`:
```python
USE_SIGMOID_ALPHA = True  # Change from False to True
```

Restart the service:
```bash
# Development
uvicorn app.main:app --reload

# Production (example with systemd)
sudo systemctl restart emotion-api
```

### Rollback to Linear Formula

Edit `api/app/services/feature_config.py`:
```python
USE_SIGMOID_ALPHA = False  # Change from True to False
```

Restart the service (same commands as above).

**Rollback time:** < 1 minute (config change + restart)

---

## Migration Checklist

### Phase 1: Pre-Deployment Validation (1-2 days)

- [ ] **Review current alpha behavior**
  - Check alpha distribution in production logs
  - Document current mean/median alpha values
  - Identify any edge cases or anomalies

- [ ] **Run tests in staging environment**
  ```bash
  cd api
  pytest tests/test_alpha_engine.py
  pytest tests/test_alpha_engine_properties.py
  pytest tests/test_alpha_formula_comparison.py
  ```

- [ ] **Verify backward compatibility**
  - Deploy with `USE_SIGMOID_ALPHA = False`
  - Confirm no regressions in API responses
  - Validate existing functionality works unchanged

- [ ] **Test sigmoid formula in staging**
  - Deploy with `USE_SIGMOID_ALPHA = True`
  - Process sample audio files
  - Review alpha values in API responses
  - Compare predictions with linear formula

- [ ] **Review hyperparameter defaults**
  - K = 50 (feedback scale)
  - τ = 0.6 (confidence threshold)
  - β = 10 (sigmoid sharpness)
  - Adjust if needed based on staging results

### Phase 2: Production Deployment (1 day)

- [ ] **Deploy with sigmoid disabled (safe default)**
  ```python
  USE_SIGMOID_ALPHA = False
  ```
  - Deploy code changes
  - Verify service health
  - Confirm no regressions

- [ ] **Enable sigmoid for monitoring**
  ```python
  USE_SIGMOID_ALPHA = True
  ```
  - Restart service
  - Monitor startup logs for "Alpha Engine: sigmoid" message
  - Verify hyperparameters are logged correctly

- [ ] **Monitor initial behavior**
  - Check API responses include new fields:
    - `alpha_data`
    - `alpha_conf`
    - `alpha_formula: "sigmoid"`
  - Review alpha distribution in logs
  - Compare with baseline from Phase 1

- [ ] **Validate predictions**
  - Process test audio samples
  - Verify emotions are reasonable
  - Check confidence scores are in expected ranges
  - Compare with linear formula predictions (if available)

### Phase 3: Monitoring and Tuning (1-2 weeks)

- [ ] **Track key metrics**
  - Alpha distribution (mean, median, std, percentiles)
  - Prediction accuracy (if ground truth available)
  - User feedback (if collected)
  - API response times

- [ ] **Identify tuning needs**
  - Is personalization too fast/slow? → Adjust K
  - Is global head over/underused? → Adjust τ
  - Are transitions too abrupt/gradual? → Adjust β

- [ ] **Apply tuning adjustments** (see Hyperparameter Tuning section)

- [ ] **Document final configuration**
  - Record optimal K, τ, β values
  - Document any edge cases discovered
  - Update runbooks with lessons learned

---

## Rollback Procedure

### When to Rollback

Consider rolling back if you observe:
- **Prediction quality degradation**: Emotions are less accurate than before
- **Unexpected alpha values**: Alpha consistently outside expected ranges (e.g., always near 0 or 1)
- **Performance issues**: API response times increase significantly (unlikely, but monitor)
- **User complaints**: Increased reports of incorrect emotion detection

### Rollback Steps

1. **Edit configuration file**
   ```bash
   cd api/app/services
   nano feature_config.py  # or vim, emacs, etc.
   ```

2. **Change flag to False**
   ```python
   # Alpha Engine Configuration (Sigmoid-Based Blending)
   USE_SIGMOID_ALPHA = False  # ← Change this line
   ALPHA_FEEDBACK_SCALE_K = 50
   ALPHA_CONFIDENCE_THRESHOLD_TAU = 0.6
   ALPHA_SIGMOID_SHARPNESS_BETA = 10
   ```

3. **Restart service**
   ```bash
   # Development
   # Stop current process (Ctrl+C) and restart
   uvicorn app.main:app --reload

   # Production (systemd example)
   sudo systemctl restart emotion-api

   # Production (Docker example)
   docker-compose restart api

   # Production (Kubernetes example)
   kubectl rollout restart deployment/emotion-api
   ```

4. **Verify rollback**
   - Check startup logs for "Alpha Engine: linear" message
   - Verify API responses no longer include `alpha_data`, `alpha_conf` fields
   - Confirm `alpha_formula` field shows "linear"
   - Test with sample audio files

5. **Monitor post-rollback**
   - Verify predictions return to baseline behavior
   - Check alpha values match pre-migration distribution
   - Confirm no errors in logs

**Expected downtime:** < 1 minute (service restart only)

**Data loss:** None (rollback is configuration-only)

---

## Hyperparameter Tuning

### Default Values

```python
ALPHA_FEEDBACK_SCALE_K = 50
ALPHA_CONFIDENCE_THRESHOLD_TAU = 0.6
ALPHA_SIGMOID_SHARPNESS_BETA = 10
```

These defaults provide balanced behavior for most use cases. Tune based on production metrics and user feedback.

### K (Feedback Scale Constant)

**What it controls:** How quickly the system transitions from global to user predictions as feedback accumulates

**Tuning signals:**

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| Users complain predictions change too quickly | Personalization too fast | Increase K (try 75 or 100) |
| Users complain system doesn't learn their preferences | Personalization too slow | Decrease K (try 30 or 40) |
| Alpha_data drops below 0.5 too early | Threshold too low | Increase K |
| Alpha_data stays near 1.0 too long | Threshold too high | Decrease K |

**Example adjustments:**
```python
# Fast personalization (aggressive learning)
ALPHA_FEEDBACK_SCALE_K = 30

# Slow personalization (conservative learning)
ALPHA_FEEDBACK_SCALE_K = 100
```

**Validation:**
- Monitor alpha_data distribution across users
- Check feedback_count at which alpha_data = 0.5
- Verify users with 50-100 feedback samples have reasonable alpha values

### τ (Tau - Confidence Threshold)

**What it controls:** The confidence level at which the system equally trusts global and user heads

**Tuning signals:**

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| Global head underused despite high confidence | Threshold too high | Decrease τ (try 0.5) |
| Global head overused despite low confidence | Threshold too low | Increase τ (try 0.7) |
| Alpha_conf < 0.5 for most predictions | Global head underconfident | Decrease τ |
| Alpha_conf > 0.5 for most predictions | Global head overconfident | Increase τ |

**Example adjustments:**
```python
# Trust global head more easily
ALPHA_CONFIDENCE_THRESHOLD_TAU = 0.5

# Require high confidence to trust global head
ALPHA_CONFIDENCE_THRESHOLD_TAU = 0.7
```

**Validation:**
- Plot alpha_conf vs global_confidence scatter plot
- Verify sigmoid center is at desired confidence level
- Check that high-confidence predictions (C_g > 0.8) have high alpha_conf

### β (Beta - Sigmoid Sharpness)

**What it controls:** How steep the transition is between trusting global vs user head

**Tuning signals:**

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| Alpha changes too abruptly with small confidence changes | Transition too sharp | Decrease β (try 5) |
| Alpha changes too gradually, unclear which head is trusted | Transition too gentle | Increase β (try 15 or 20) |
| Want more decisive switching behavior | Need sharper transition | Increase β |
| Want smoother blending across confidence range | Need gentler transition | Decrease β |

**Example adjustments:**
```python
# Gentle, gradual transition
ALPHA_SIGMOID_SHARPNESS_BETA = 5

# Sharp, decisive transition
ALPHA_SIGMOID_SHARPNESS_BETA = 20
```

**Validation:**
- Plot alpha_conf vs global_confidence curve
- Verify transition steepness matches desired behavior
- Check that confidence changes of ±0.1 produce appropriate alpha changes

### Tuning Workflow

1. **Establish baseline**
   - Deploy with defaults (K=50, τ=0.6, β=10)
   - Collect 1-2 weeks of production data
   - Document alpha distribution, prediction accuracy, user feedback

2. **Identify primary issue**
   - Personalization speed → Adjust K first
   - Global head usage → Adjust τ first
   - Transition behavior → Adjust β first

3. **Make incremental changes**
   - Change one parameter at a time
   - Use 20-30% adjustments (e.g., K: 50 → 60 or 50 → 40)
   - Deploy and monitor for 2-3 days

4. **Validate impact**
   - Compare alpha distribution before/after
   - Check prediction accuracy metrics
   - Review user feedback

5. **Iterate**
   - If improvement, consider further tuning
   - If degradation, revert and try different parameter
   - Document final configuration

### Monitoring Queries

Use these log queries to monitor alpha behavior:

```bash
# Check alpha distribution
grep "Alpha computation" /var/log/emotion-api.log | \
  awk '{print $NF}' | \
  sort -n | \
  awk '{sum+=$1; sumsq+=$1*$1} END {print "Mean:", sum/NR, "Std:", sqrt(sumsq/NR - (sum/NR)^2)}'

# Find alpha values for specific user
grep "user_id=USER123" /var/log/emotion-api.log | \
  grep "Alpha computation"

# Check alpha_data distribution
grep "alpha_data" /var/log/emotion-api.log | \
  awk -F'data=' '{print $2}' | \
  awk -F',' '{print $1}' | \
  sort -n

# Check alpha_conf distribution
grep "alpha_conf" /var/log/emotion-api.log | \
  awk -F'conf=' '{print $2}' | \
  awk -F',' '{print $1}' | \
  sort -n
```

---

## Expected Behavior Changes

### Key Differences from Linear Formula

| Aspect | Linear Formula | Sigmoid Formula | Impact |
|--------|----------------|-----------------|--------|
| **New users (N=0)** | Always alpha=1.0 | Responds to confidence | More nuanced for new users |
| **Feedback decay** | Linear | Exponential | Faster initial decay |
| **Confidence response** | Linear | S-curve | Smoother transitions |
| **Combination** | Additive | Multiplicative | Both factors must agree |
| **Bounds** | Hard clamp [0.3, 1.0] | Natural [0, 1] | Wider alpha range |

### Concrete Examples

#### Example 1: New User, Low Confidence

**Scenario:** User uploads first audio, global head predicts "happy" with 50% confidence

**Linear:**
```
alpha = 1.0 (hardcoded for N < 20)
Final prediction = 100% global head
```

**Sigmoid:**
```
alpha_data = 1.0 (no feedback)
alpha_conf = 0.27 (low confidence)
alpha = 0.27
Final prediction = 27% global, 73% user head
```

**Impact:** Sigmoid is more conservative with low-confidence predictions, even for new users

#### Example 2: Medium Feedback, High Confidence

**Scenario:** User has 50 feedback samples, global head predicts "sad" with 90% confidence

**Linear:**
```
alpha = 0.5 + 0.27 - 0.1 = 0.67
Final prediction = 67% global, 33% user
```

**Sigmoid:**
```
alpha_data = 0.5 (at threshold)
alpha_conf = 0.95 (high confidence)
alpha = 0.48
Final prediction = 48% global, 52% user
```

**Impact:** Sigmoid favors user head more, even with high global confidence

#### Example 3: High Feedback, Medium Confidence

**Scenario:** User has 100 feedback samples, global head predicts "angry" with 70% confidence

**Linear:**
```
alpha = 0.5 + 0.21 - 0.2 = 0.51
Final prediction = 51% global, 49% user
```

**Sigmoid:**
```
alpha_data = 0.33 (lots of feedback)
alpha_conf = 0.73 (medium confidence)
alpha = 0.24
Final prediction = 24% global, 76% user
```

**Impact:** Sigmoid strongly favors user head with lots of feedback

### Visual Comparison

**Alpha vs Feedback Count (C_g = 0.7):**
```
Feedback  | Linear | Sigmoid | Difference
----------|--------|---------|------------
0         | 1.00   | 0.73    | -0.27 (sigmoid lower)
10        | 1.00   | 0.69    | -0.31
20        | 0.71   | 0.65    | -0.06
50        | 0.61   | 0.37    | -0.24 (sigmoid much lower)
100       | 0.51   | 0.24    | -0.27
200       | 0.51   | 0.15    | -0.36 (sigmoid strongly favors user)
```

**Alpha vs Confidence (N = 50):**
```
Confidence | Linear | Sigmoid | Difference
-----------|--------|---------|------------
0.3        | 0.49   | 0.07    | -0.42 (sigmoid much lower)
0.5        | 0.55   | 0.14    | -0.41
0.6        | 0.58   | 0.25    | -0.33
0.7        | 0.61   | 0.37    | -0.24
0.9        | 0.67   | 0.48    | -0.19 (sigmoid still lower)
```

### User-Facing Changes

**What users might notice:**
1. **Faster personalization**: System adapts to user preferences more quickly
2. **More conservative with uncertainty**: Low-confidence predictions rely more on user history
3. **Smoother transitions**: No abrupt changes in prediction behavior
4. **Better handling of edge cases**: No hard cutoffs at 20 feedback samples

**What users should NOT notice:**
1. API response format (same fields, new optional fields added)
2. Prediction quality (should be same or better)
3. Response times (< 1ms overhead)

---

## Monitoring and Validation

### Key Metrics to Track

1. **Alpha Distribution**
   - Mean, median, std deviation
   - Percentiles (p10, p25, p50, p75, p90)
   - Compare with baseline from linear formula

2. **Alpha Components**
   - alpha_data distribution (should decrease with feedback)
   - alpha_conf distribution (should correlate with confidence)
   - Correlation between alpha_data and alpha_conf

3. **Prediction Quality**
   - Emotion accuracy (if ground truth available)
   - Confidence calibration (predicted confidence vs actual accuracy)
   - User feedback (if collected)

4. **Performance**
   - API response time (should be < 1ms overhead)
   - CPU usage (should be negligible)
   - Memory usage (should be unchanged)

### Validation Checklist

- [ ] **Startup logs show correct formula**
  ```
  INFO: Alpha Engine: sigmoid (K=50, tau=0.6, beta=10)
  ```

- [ ] **API responses include new fields**
  ```json
  {
    "alpha_data": 0.67,
    "alpha_conf": 0.73,
    "alpha_formula": "sigmoid"
  }
  ```

- [ ] **Alpha values are in expected ranges**
  - alpha_data: (0, 1], decreases with feedback
  - alpha_conf: (0, 1), increases with confidence
  - alpha: (0, 1), product of alpha_data and alpha_conf

- [ ] **No errors in logs**
  - No NaN or infinity values
  - No exceptions in alpha computation
  - No performance warnings

- [ ] **Predictions are reasonable**
  - Emotions match audio content
  - Confidence scores are calibrated
  - No systematic biases

### Troubleshooting

**Issue:** Alpha values are always near 1.0

**Possible causes:**
- K is too high (slow personalization)
- τ is too low (easy to trust global head)
- Users have very little feedback

**Solutions:**
- Decrease K to speed up personalization
- Increase τ to require higher confidence
- Verify feedback collection is working

---

**Issue:** Alpha values are always near 0.0

**Possible causes:**
- K is too low (fast personalization)
- τ is too high (hard to trust global head)
- Global head has low confidence

**Solutions:**
- Increase K to slow down personalization
- Decrease τ to trust global head more easily
- Check global head calibration

---

**Issue:** Predictions are worse than before

**Possible causes:**
- Hyperparameters not tuned for your data
- Global head confidence is poorly calibrated
- User head training is insufficient

**Solutions:**
- Revert to linear formula (rollback)
- Tune hyperparameters based on production data
- Improve global head confidence calibration
- Collect more user feedback for training

---

**Issue:** API response times increased

**Possible causes:**
- Logging level set too high (INFO/DEBUG)
- Numpy not optimized
- Other unrelated performance issues

**Solutions:**
- Set logging to WARNING in production
- Verify numpy is using optimized BLAS
- Profile API to identify bottleneck

---

## FAQ

### Q: Can I switch between formulas without restarting?

**A:** No, the `USE_SIGMOID_ALPHA` flag is read at startup. You must restart the service for changes to take effect.

### Q: Will switching formulas affect existing user data?

**A:** No, the formula only affects how predictions are blended. User feedback, embeddings, and model weights are unchanged.

### Q: Can I use different hyperparameters for different users?

**A:** Not currently. Hyperparameters are global. Future enhancements may support per-user tuning.

### Q: How do I know if sigmoid is better than linear?

**A:** Compare these metrics over 1-2 weeks:
- Prediction accuracy (if ground truth available)
- User feedback (if collected)
- Alpha distribution (should be more intuitive)
- User complaints (should decrease)

### Q: What if I want to tune hyperparameters in production?

**A:** Edit `feature_config.py`, change K/τ/β values, and restart the service. Monitor for 2-3 days before further adjustments.

### Q: Is there a way to A/B test the formulas?

**A:** Not built-in. You would need to implement user-level feature flags and route requests accordingly. Consider testing in staging first.

### Q: What's the performance overhead of sigmoid?

**A:** < 1ms per prediction. The sigmoid computation uses numpy's optimized exp() function and is negligible compared to model inference.

### Q: Can I see alpha values in production logs?

**A:** Yes, set logging level to DEBUG. Alpha values are logged for each prediction:
```
DEBUG: Alpha computation: data=0.67, conf=0.73, final=0.49 (N=50, C_g=0.70)
```

---

## Support and Resources

### Documentation
- **Technical README**: `api/README.md` - Detailed alpha engine documentation
- **Design Document**: `.kiro/specs/alpha-engine-refinement/design.md` - Mathematical formulation
- **Requirements**: `.kiro/specs/alpha-engine-refinement/requirements.md` - Acceptance criteria

### Code References
- **Alpha Engine**: `api/app/services/alpha_engine.py` - Core implementation
- **Configuration**: `api/app/services/feature_config.py` - Hyperparameters
- **Tests**: `api/tests/test_alpha_engine*.py` - Unit and property tests

### Contact
For issues or questions:
1. Check logs for error messages
2. Review this migration guide
3. Consult technical documentation
4. Open an issue on the repository

---

## Appendix: Configuration File Reference

### Complete Configuration Block

```python
# api/app/services/feature_config.py

# Alpha Engine Configuration (Sigmoid-Based Blending)
USE_SIGMOID_ALPHA = False  # Set to True to enable sigmoid formula
ALPHA_FEEDBACK_SCALE_K = 50  # Feedback count at which alpha_data = 0.5
ALPHA_CONFIDENCE_THRESHOLD_TAU = 0.6  # Confidence threshold for sigmoid center
ALPHA_SIGMOID_SHARPNESS_BETA = 10  # Sigmoid transition sharpness
```

### Parameter Constraints

- `USE_SIGMOID_ALPHA`: Boolean (True or False)
- `ALPHA_FEEDBACK_SCALE_K`: Float > 0 (typical range: 25-100)
- `ALPHA_CONFIDENCE_THRESHOLD_TAU`: Float in (0, 1) (typical range: 0.5-0.7)
- `ALPHA_SIGMOID_SHARPNESS_BETA`: Float > 0 (typical range: 5-20)

### Validation

The system validates hyperparameters at startup. Invalid values will cause startup failure with clear error messages.

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Applies to:** Alpha Engine Refinement (Sigmoid-Based Blending)
