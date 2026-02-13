# Development Environment Testing Checklist

## Task 7.2: Test in Development Environment

This checklist guides testing of the alpha engine in a development environment before production deployment.

---

## Prerequisites

- [ ] Development environment is running
- [ ] All tests pass locally (`pytest`)
- [ ] API server can start without errors

---

## Phase 1: Baseline Testing (Linear Formula)

**Goal:** Verify no regressions with USE_SIGMOID_ALPHA=False

### 1.1 Configuration Check
```bash
# Verify USE_SIGMOID_ALPHA is False in feature_config.py
grep "USE_SIGMOID_ALPHA" api/app/services/feature_config.py
```

Expected: `USE_SIGMOID_ALPHA = False`

### 1.2 Start API Server
```bash
cd api
python -m app.main
```

**Check startup logs:**
- [ ] Log shows: "Alpha Engine: linear"
- [ ] No errors during startup
- [ ] Models load successfully

### 1.3 Test API Endpoints

**Test 1: New user (N=0)**
```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_audio.wav" \
  -F "user_id=test_user_new"
```

Expected response:
- `blend_weight`: 1.0 (for N < 20)
- `alpha_formula`: "linear"
- `alpha_data`: null
- `alpha_conf`: null

**Test 2: Medium feedback user (N=50)**
```bash
# Simulate user with 50 feedback samples
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_audio.wav" \
  -F "user_id=test_user_medium"
```

Expected response:
- `blend_weight`: Between 0.3 and 1.0
- `alpha_formula`: "linear"
- Predictions match previous behavior

### 1.4 Verify Logs
```bash
# Check logs for alpha computation
tail -f logs/api.log | grep -i alpha
```

Expected:
- No sigmoid-related logs
- Linear formula being used
- No errors or warnings

---

## Phase 2: Sigmoid Testing (New Formula)

**Goal:** Verify new sigmoid behavior works correctly

### 2.1 Enable Sigmoid Formula
```python
# In api/app/services/feature_config.py
USE_SIGMOID_ALPHA = True
```

### 2.2 Restart API Server
```bash
cd api
python -m app.main
```

**Check startup logs:**
- [ ] Log shows: "Alpha Engine: sigmoid (K=50, tau=0.6, beta=10)"
- [ ] Hyperparameters logged correctly
- [ ] No errors during startup

### 2.3 Test API Endpoints with Sigmoid

**Test 1: New user with low confidence**
```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_audio_low_conf.wav" \
  -F "user_id=test_user_sigmoid_1"
```

Expected response:
- `alpha_formula`: "sigmoid"
- `alpha_data`: ~1.0 (N=0)
- `alpha_conf`: < 0.5 (low confidence)
- `blend_weight`: < 0.5 (alpha_data × alpha_conf)

**Test 2: New user with high confidence**
```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_audio_high_conf.wav" \
  -F "user_id=test_user_sigmoid_2"
```

Expected response:
- `alpha_formula`: "sigmoid"
- `alpha_data`: ~1.0 (N=0)
- `alpha_conf`: > 0.5 (high confidence)
- `blend_weight`: > 0.5

**Test 3: Medium feedback user**
```bash
# User with ~50 feedback samples
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_audio.wav" \
  -F "user_id=test_user_sigmoid_medium"
```

Expected response:
- `alpha_formula`: "sigmoid"
- `alpha_data`: ~0.5 (N≈50)
- `alpha_conf`: varies with confidence
- `blend_weight`: Lower than linear formula (favors user head more)

### 2.4 Verify Debug Logs
```bash
# Enable DEBUG logging and check alpha components
tail -f logs/api.log | grep "Alpha computation"
```

Expected log format:
```
Alpha computation: data=0.500, conf=0.731, final=0.366 (N=50, C_g=0.700)
```

Verify:
- [ ] All three components logged (data, conf, final)
- [ ] Values are in expected ranges
- [ ] Multiplicative relationship holds: final ≈ data × conf

---

## Phase 3: Comparison Testing

**Goal:** Compare sigmoid vs linear predictions side-by-side

### 3.1 Collect Baseline Data (Linear)
```bash
# Set USE_SIGMOID_ALPHA = False
# Run test suite and save results
pytest api/tests/test_alpha_formula_comparison.py -v > linear_results.txt
```

### 3.2 Collect Sigmoid Data
```bash
# Set USE_SIGMOID_ALPHA = True
# Run same tests and save results
pytest api/tests/test_alpha_formula_comparison.py -v > sigmoid_results.txt
```

### 3.3 Compare Results
```bash
# Review comparison document
cat api/tests/ALPHA_FORMULA_COMPARISON.md
```

Verify:
- [ ] Sigmoid favors user head more aggressively
- [ ] Transitions are smoother (no discontinuities)
- [ ] Behavior matches design document predictions

---

## Phase 4: Real Audio Testing

**Goal:** Test with actual audio samples from different scenarios

### 4.1 Test Scenarios

**Scenario 1: Happy speech (high confidence)**
- Record or use sample happy audio
- Test with new user (N=0)
- Test with medium feedback user (N=50)
- Compare linear vs sigmoid predictions

**Scenario 2: Ambiguous speech (low confidence)**
- Use audio with mixed emotions
- Test with new user (N=0)
- Test with medium feedback user (N=50)
- Verify sigmoid responds to low confidence

**Scenario 3: User with lots of feedback (N=100+)**
- Simulate user with 100+ feedback samples
- Test with various audio samples
- Verify sigmoid strongly favors user head

### 4.2 Prediction Quality Check
- [ ] Predictions are reasonable (match expected emotions)
- [ ] Confidence scores are calibrated (not all 0.99 or 0.01)
- [ ] Blending behavior makes intuitive sense
- [ ] No crashes or errors

---

## Phase 5: Performance Testing

**Goal:** Verify sigmoid adds minimal overhead

### 5.1 Run Performance Benchmarks
```bash
pytest api/tests/test_alpha_engine_performance.py -v
```

Expected:
- [ ] Sigmoid computation < 1ms per call
- [ ] No significant difference from linear formula
- [ ] No memory leaks or performance degradation

### 5.2 Load Testing (Optional)
```bash
# Use tool like Apache Bench or locust
ab -n 1000 -c 10 http://localhost:8000/api/classify
```

Verify:
- [ ] Response times acceptable
- [ ] No errors under load
- [ ] Sigmoid doesn't cause bottlenecks

---

## Phase 6: Rollback Testing

**Goal:** Verify instant rollback capability

### 6.1 Switch Back to Linear
```python
# In api/app/services/feature_config.py
USE_SIGMOID_ALPHA = False
```

### 6.2 Restart and Verify
```bash
python -m app.main
```

Expected:
- [ ] Startup log shows "Alpha Engine: linear"
- [ ] API works immediately
- [ ] No data loss or corruption
- [ ] Predictions revert to linear behavior

---

## Success Criteria

All checkboxes above should be checked before proceeding to production deployment (Task 7.4).

**Key Validations:**
- ✓ No regressions with linear formula
- ✓ Sigmoid formula works correctly
- ✓ All three alpha components logged properly
- ✓ Predictions are reasonable and intuitive
- ✓ Performance is acceptable (< 1ms overhead)
- ✓ Rollback works instantly

---

## Notes

- Keep USE_SIGMOID_ALPHA=False as default until production validation complete
- Document any unexpected behavior or edge cases discovered
- Save test results for comparison with production metrics
- If any issues found, fix before proceeding to Task 7.4

---

## Next Steps

Once all checks pass:
1. Document findings in `DEV_ENVIRONMENT_TEST_RESULTS.md`
2. Mark Task 7.2 as complete
3. Proceed to Task 7.4 (Production Deployment)
