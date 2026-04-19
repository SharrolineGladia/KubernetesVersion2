# COMPREHENSIVE HANDOFF PROMPT - Anomaly Detection & Recovery System

## PROJECT OVERVIEW

Building an **intelligent anomaly detection and automated recovery system** for microservices on Kubernetes. The system detects anomalies in real-time, explains WHY they happen, and shows WHAT-IF scenarios for prevention.

**Current Status:** Core pipeline complete, enhancement phase required

---

## COMPLETED COMPONENTS ✅

### 1. EWMA Online Detector (`online_detector/`)

- **Status:** COMPLETE and working
- **What it does:** Continuously monitors 3 EWMA channels:
  - Resource Saturation (CPU + Memory)
  - Pressure Buildup (Memory pressure + Error rate)
  - Performance Degradation (Response time + Queue depth)
- **Output:** Triggers snapshot when system reaches CRITICAL state
- **Integration:** Runs continuously, feeds frozen snapshots to ML detector

**Files:**

- `online_detector/main.py` - Main loop
- `online_detector/detector.py` - EWMA logic
- `online_detector/config.py` - Configuration

---

### 2. ML Detector (XGBoost) (`ml_detector/scripts/`)

- **Status:** COMPLETE, trained model ready
- **What it does:** Classifies anomalies into 4 types:
  - `cpu_spike` - High CPU utilization
  - `memory_leak` - Memory pressure/growth
  - `service_crash` - High error rates
  - `normal` - No anomaly
- **Feature Space:** 27 features extracted from service metrics
- **Model Performance:** 78-82% accuracy across anomaly types
- **Files:**
  - `dual_feature_detector.py` - Feature extraction and prediction
  - `anomaly_detector_scaleinvariant.pkl` - Trained model

---

### 3. Root Cause Analysis (RCA) (`ml_detector/scripts/explainability_layer.py`)

- **Status:** COMPLETE
- **What it does:** Identifies which service caused the anomaly
- **Output:**
  - Root cause service name
  - Contributing factors (which metrics spiked)
  - Severity level
  - RCA recommendations
- **Integration:** Runs after XGBoost classification

---

### 4. Counterfactual Analysis (`ml_detector/scripts/counterfactual_analyzer.py`)

- **Status:** ~60% COMPLETE - NEEDS ENHANCEMENT
- **Current Capability:**
  - ✅ Tests top-5 most important features
  - ✅ Generates multiple prevention scenarios
  - ✅ Computes minimum change needed for each
  - ✅ Stores alternative actions
  - ❌ **MISSING:** Shows predicted outcomes for each scenario
  - ❌ **MISSING:** Comparison table/ranking
  - ❌ **MISSING:** "What-if system state" after each change

**Current Output Example:**

```
🎯 Primary Cause: cpu_utilization_mean
   Current: 0.85 → Target: 0.40 (-52.6%)

🔄 Alternative Prevention Strategies:
   1. memory_utilization_mean: 0.45 → 0.28 (-37.8%)
   2. system_stress_index: 0.72 → 0.40 (-44.4%)
```

**What it SHOULD show:**

```
🎯 SCENARIO 1: Reduce CPU by 52.6%
   New state: CPU 0.40, Memory 0.45, Errors 0.05
   Prediction: NORMAL (87% confidence)
   Status: ✅ PREVENTS ANOMALY

🎯 SCENARIO 2: Reduce Memory by 37.8%
   New state: CPU 0.85, Memory 0.28, Errors 0.05
   Prediction: CPU_SPIKE (75% confidence)
   Status: ❌ STILL ANOMALOUS

🎯 SCENARIO 3: Reduce Stress by 44.4%
   New state: CPU 0.65, Memory 0.38, Errors 0.04
   Prediction: NORMAL (92% confidence)
   Status: ✅ PREVENTS + BEST SCORE
```

---

### 5. Demo Scripts (COMPLETE)

- `scripts/demos/demo_counterfactual_output.py` - 3 anomaly scenarios with counterfactuals
- `scripts/demos/demo_counterfactual_showcase.py` - **USE THIS FOR SLIDES** - polished single output
- `scripts/demos/demo_full_pipeline.py` - Complete EWMA → XGBoost → RCA → Counterfactuals flow

---

## CURRENT ARCHITECTURE

```
┌─────────────────────────────────────────────┐
│         Service Metrics (Prometheus)        │
└────────────────────┬────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  EWMA Online Detector  │
        │  (3 channels)          │
        │  State: NORMAL→        │
        │         STRESSED→      │
        │         CRITICAL       │
        └────────────┬───────────┘
                     │ (when CRITICAL)
                     ▼
        ┌────────────────────────┐
        │  Snapshot Frozen       │
        │  27 service metrics    │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  XGBoost Classifier    │
        │  Output: 4 anomaly     │
        │  types + confidence    │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  RCA Module            │
        │  Identify root cause   │
        │  service + factors     │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Counterfactual Analysis   │
        │  (ENHANCEMENT NEEDED)      │
        │  What prevents anomaly?    │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Recovery Orchestrator     │
        │  (NEED TO BUILD)           │
        │  Execute recovery actions  │
        └────────────────────────────┘
```

---

## HIGH PRIORITY TASKS 🔴

### **TASK 1: Enhance Counterfactual Analysis** (Currently In Progress)

**Priority:** CRITICAL - Needed before handoff

**What needs to be done:**

1. **Modify `counterfactual_analyzer.py`**
   - For each alternative action (feature change), predict what the new system state would be
   - Show the predicted anomaly class and confidence for each scenario
   - Implement comparison/ranking (which scenario best prevents the anomaly?)

2. **Output Format Enhancement**
   - Create a comparison table showing:
     - Scenario name
     - Feature changes
     - New predicted anomaly type
     - Confidence score
     - Feasibility percentage
     - Risk level (if change is too aggressive)

3. **Actionable Scoring**
   - Score each scenario (0-100)
   - Factors: Feasibility, Confidence, Simplicity of implementation
   - Highlight the TOP recommended scenario

**Expected Output:**

```
╔════════════════════════════════════════════════════════════╗
║           COUNTERFACTUAL WHAT-IF SCENARIOS                 ║
╚════════════════════════════════════════════════════════════╝

📊 SCENARIO COMPARISON:

  SCENARIO          CHANGE              PREDICTION      CONFIDENCE  RECOMMENDED
  ─────────────────────────────────────────────────────────────────────────────
  ✅ Scale CPU      CPU 85%→40%         NORMAL          87%         ⭐ BEST
  ❌ Reduce Memory  Memory 45%→28%      CPU_SPIKE       75%
  ✅ Stress Index   Stress 72%→40%      NORMAL          92%         ⭐ BEST ALT

📋 Recommendation: Scale horizontally (add replicas) to reduce CPU

Feasibility: 95% | Implementation: kubectl scale deployment web-api --replicas=3
```

**Estimated Work:** 2-3 hours

---

### **TASK 2: Transparent Decision-Making - Recovery Action Scoring** (NEXT)

**Priority:** HIGH - Needed for automated recovery

**What needs to be built:**

1. **Recovery Action Generator**
   - Given the counterfactual scenario and root cause, generate concrete K8s recovery actions
   - Examples:
     - `kubectl scale deployment X --replicas=+2`
     - `kubectl set resources deployment X --limits=cpu=2,memory=4Gi`
     - `kubectl restart deployment X`
     - `kubectl apply -f optimized-config.yaml`

2. **Action Scoring System**
   - Score each action on:
     - **Effectiveness:** Will this solve the problem? (0-1)
     - **Safety:** Will this cause downtime? (0-1)
     - **Speed:** How fast can it execute? (seconds)
     - **Cost:** Resource impact (0-1)
     - **Simplicity:** How easy to implement? (0-1)

3. **Action Comparison UI**
   - Show multiple recovery action options
   - Rank by combined score
   - Show tradeoffs between speed vs safety vs cost

**Files to Create:**

- `recovery_orchestrator/action_generator.py`
- `recovery_orchestrator/action_scorer.py`
- Update `online_detector/main.py` to call recovery actions

**Estimated Work:** 3-4 hours

---

## MEDIUM PRIORITY TASKS 🟡

### **TASK 3: Recovery Action Executor**

- Execute the recommended recovery action on Kubernetes cluster
- Handle rollback if action fails
- Log all actions for audit
- **Files:** `recovery_orchestrator/executor.py`
- **Estimated Work:** 2-3 hours

### **TASK 4: Feedback Loop & Learning**

- Track which counterfactuals actually prevented anomalies
- Update counterfactual recommendations based on outcomes
- Store historical data for pattern learning
- **Files:** `recovery_orchestrator/feedback_collector.py`
- **Estimated Work:** 2-3 hours

### **TASK 5: Monitoring Dashboard**

- Real-time display of:
  - EWMA channel states
  - Detected anomalies
  - RCA results
  - Counterfactual scenarios
  - Recovery actions executed
  - System health after recovery
- **Tech:** Grafana/Prometheus or custom Flask dashboard
- **Estimated Work:** 4-6 hours

---

## LOW PRIORITY TASKS 🟢

### **TASK 6: Multi-Service Dependencies**

- Currently: Each service analyzed independently
- Future: Consider service-to-service dependencies
- Example: If service A fails, what happens to B?

### **TASK 7: Historical Analysis**

- Store past anomalies + counterfactuals + outcomes
- Build analytics dashboard
- Identify recurring patterns

### **TASK 8: ML Model Improvement**

- Retrain on more data
- Add ensemble methods
- Hyperparameter tuning

---

## FILE STRUCTURE

```
demo/
├── online_detector/
│   ├── main.py ✅ (Runs EWMA loop, triggers snapshots)
│   ├── detector.py ✅ (EWMA channels logic)
│   └── config.py ✅ (Configuration)
│
├── ml_detector/
│   ├── models/
│   │   └── anomaly_detector_scaleinvariant.pkl ✅ (Trained model)
│   ├── scripts/
│   │   ├── dual_feature_detector.py ✅ (Feature extraction + prediction)
│   │   ├── counterfactual_analyzer.py ⚠️ (NEEDS ENHANCEMENT)
│   │   ├── explainability_layer.py ✅ (RCA)
│   │   └── [various analysis scripts]
│   └── results/
│
├── recovery_orchestrator/ ❌ (TO BE BUILT)
│   ├── action_generator.py
│   ├── action_scorer.py
│   ├── executor.py
│   └── feedback_collector.py
│
├── escalation_optimizer/ ❌ (TO BE BUILT - for critical situations)
│
├── demo_*.py ✅ (Demo/showcase scripts)
└── test_*.py ✅ (Test scripts)
```

---

## HOW TO RUN CURRENT SYSTEM

```bash
# 1. View counterfactual output (for slides)
python scripts/demos/demo_counterfactual_showcase.py

# 2. See complete pipeline
python scripts/demos/demo_full_pipeline.py

# 3. Run live detector (generates synthetic anomalies)
python online_detector/main.py

# 4. Run tests
python scripts/tests/test_counterfactual_integration.py
```

---

## KEY INSIGHTS FOR NEXT AGENT 👨‍💼

1. **Counterfactual Feature Gap:** Current implementation shows alternatives but NOT their predicted outcomes. User correctly pointed out that true counterfactuals should show "what would happen if we made change X?"

2. **Model Uses 27 Features:** All normalized to [0, 1] range. Important for scaling predictions.

3. **Class Order:** XGBoost classes are alphabetical: `['cpu_spike', 'memory_leak', 'normal', 'service_crash']`

4. **EWMA Sensitivity:** Currently tuned for Kubernetes workloads. May need adjustment for different environments.

5. **RCA Limitations:** Currently single-service root cause. May need multi-service correlation analysis for complex scenarios.

6. **No Execution Layer Yet:** System detects, explains, and suggests - but doesn't execute. Need to build safe executor with rollback capability.

7. **Model Training:** XGBoost model was trained on simulated data. Needs real production data for better accuracy.

---

## CRITICAL SUCCESS FACTORS ✓

- [ ] Counterfactuals show predicted outcomes (TASK 1)
- [ ] Recovery actions are safe + tested (TASK 3)
- [ ] System can rollback failed actions (TASK 3)
- [ ] Audit log of all actions taken (TASK 3)
- [ ] Dashboard shows system health (TASK 5)
- [ ] Feedback loop exists (TASK 4)
- [ ] Real production testing (ONGOING)

---

## QUESTION FOR NEXT AGENT

**What is your priority focus?**

A) **Enhancement Path (Recommended):**

1.  Complete Counterfactuals (TASK 1)
2.  Build Recovery Orchestrator (TASK 2-3)
3.  Add Feedback Loop (TASK 4)
    → Gives complete end-to-end system

B) **Monitoring Path:**

1.  Build Dashboard (TASK 5)
2.  Historical Analysis (TASK 7)
    → Gives visibility but no automation

C) **ML Path:**

1.  Retrain model on more data
2.  Improve accuracy
    → Better predictions but slower execution

D) **Production Path:**

1.  Deploy executor to real Kubernetes
2.  Add safety checks + rollbacks
    → Gets system live faster

**Recommendation:** Go with **A (Enhancement Path)** to complete the system before production deployment.

---

## GIT STATUS

**Modified Files:**

- `ml_detector/scripts/dual_feature_detector.py`
- `online_detector/config.py`
- `online_detector/detector.py`
- `online_detector/main.py`

**New Files (Untracked):**

- `demo_counterfactual_*.py` (demo scripts)
- `test_*.py` (test scripts)
- `counterfactual_analyzer.py`
- `recovery_orchestrator/` (folder - needs work)
- `escalation_optimizer/` (folder - empty)
- `results/` (analysis outputs)

**Ready to commit:** Once TASK 1 is completed

---

## NEXT STEPS

1. Hand off to agent
2. Have agent focus on **TASK 1 (Enhance Counterfactuals)**
3. Once complete, move to **TASK 2 (Recovery Actions)**
4. Test, iterate, repeat

**Estimated Total Time to Production Ready:** 8-12 hours (Tasks 1-5)

---

**Last Updated:** 2026-03-26
**Current Phase:** Enhancement (Counterfactuals)
**Status:** Ready for handoff ✅
