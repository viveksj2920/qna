# Predefined Subtopic Extraction — Test Results v2

Hi Pauline,

Thanks for the feedback! I've made the changes you suggested and re-ran the tests. Here's what's updated:

## Changes Based on Your Feedback

1. **Stricter matching** — The LLM now returns an empty list `[]` when no predefined subtopic clearly fits, instead of forcing a poor match. This will help us identify missing subtopics.
2. **Entity extraction now working** — Drug names, plan names, pharmacy names, hospital names, and dental plan names are being extracted and shown alongside the predefined subtopics.

## Gym Question Fix

"How much money for the gym?" now returns `[]` instead of `gym membership eligibility`. You're right — there's no gym cost subtopic in the current list. If you'd like to add one (e.g., "gym benefit cost" under transportation, food and gym benefits), I can include it.

## Entity Extraction Results (20 test questions)

### Drug Names
Dosages, salt forms, and brand qualifiers are automatically stripped to the base/generic name:

| Question | Subtopic | Drug Extracted |
|----------|----------|----------------|
| Is my zolpidem 10mg tablet covered? | specific medication coverage | **zolpidem** |
| I take metformin hydrochloride 500mg daily | specific medication coverage | **metformin** |
| Can you check if Eliquis is on the formulary? | specific medication coverage | **eliquis** |
| My doctor prescribed ambien (zolpidem), copay? | prescription drug copays, specific medication coverage | **zolpidem** |

### Plan Names
Plan names are extracted and fuzzy-matched against the 950 official plan list:

| Question | Subtopic | Plan Extracted |
|----------|----------|----------------|
| Compare UnitedHealthcare Dual Complete with my plan | specific plan comparison | **UnitedHealthcare Dual Complete** |
| Switch to Wellcare Value Script plan | switching plan types | **Wellcare Value Script** |
| When does AARP Medicare Advantage from UHC FL-0021 PPO start? | plan effective/start date | **AARP Medicare Advantage from UHC FL-0021 (PPO)** |
| What is the premium for Medica Prime Solution? | *(empty — no exact subtopic fit)* | **Medica Prime Solution** |

### Pharmacy / Hospital / Dental Plan Names

| Question | Subtopic | Entity Extracted |
|----------|----------|------------------|
| Is CVS Pharmacy on Oak Street in my network? | provider network status check | **CVS Pharmacy on Oak Street** |
| Fill prescription at Walgreens? | pharmacy change request, preferred pharmacy inquiry | **Walgreens** |
| Is Mayo Clinic in-network? | facility network status | **Mayo Clinic** |
| Referral to Cleveland Clinic | referral requirement clarification | **Cleveland Clinic** |
| Does my Delta Dental PPO cover root canals? | comprehensive dental coverage, major dental services coverage | **Delta Dental PPO** |

### Empty Subtopic Results (helps identify gaps)

| Question | Topic | Subtopic |
|----------|-------|----------|
| How much money for the gym? | transportation, food and gym benefits | `[]` |
| What is the monthly premium for Medica Prime Solution? | plan costs | `[]` |

These empty results suggest potential new subtopics to add (e.g., gym benefit cost, plan premium inquiry).

## Known Minor Issues

1. **"AARP Medicare Advantage Choice Plan 1"** — The fuzzy matcher matched this to "Plan A" (score 80) because "Choice Plan 1" doesn't exist in the official plan list. The actual list has names like "AARP Medicare Advantage from UHC AL-0001 (HMO-POS)". This is expected behavior — it only matches when there's a close match.
2. **"CVS Pharmacy on Oak Street"** — The LLM included the street name in the entity. We can add post-processing to strip addresses if needed.

## Next Steps

- Could you review and flag any incorrect assignments?
- Any subtopics you'd like to add based on the empty `[]` results?
- Should I run on a larger batch of real conversation data (~100-200 records)?

Attached files:
1. **test_entity_output.csv** — Entity extraction test results (20 questions)
2. **questions_output_v2.csv** — Original 26 questions re-run with stricter matching
3. **test_entity_questions.csv** — The input questions used

Thanks,
Vivek
