# Predefined Subtopic Extraction — Test Results

Hi Pauline,

The predefined subtopic extraction is implemented and tested. Here's the before/after comparison:

## Before (old system)
- LLM generated subtopics freely → `['medication']`, `['U card']`, `['monthly premium']`, `['Florida Blue']`, `['NA']`
- Entity names mixed into subtopics (doctor names, plan names, drug names treated as subtopics)
- ~300K unique subtopics with many duplicates

## After (new system)
- LLM selects only from the 420 predefined subtopics
- Entity extraction handled separately (drug names normalized, plan names fuzzy-matched against 950 plan list)
- Multi-subtopic selection working (e.g., "Can I use dental coverage elsewhere?" → `dental provider network` + `in network vs out of network dental`)

## Test Run Results (26 questions)

| Question | Topic | Subtopics (new) |
|----------|-------|-----------------|
| How can I use my U card? | general plan information | spending cards and allowances |
| Can I use dental coverage outside US? | dental | dental provider network, in network vs out of network dental |
| Can you check if my meds are covered? | prescription drug coverage | specific medication coverage |
| How much for the gym? | transportation, food and gym | gym membership eligibility |
| Doctor dropped by Florida Blue, help? | providers | keeping current provider, provider changes or leaving network |
| When does this start? Get my card. | enrollment | plan effective/start date, member id card delivery status |
| Are you taking any medication? | others | health status and assessment |
| I have a Medigap policy, what's the 2025 cost? | plan costs | plan cost comparison |
| Enroll in Medicare Advantage, which is better? | plan comparison | specific plan comparison |
| Currently enrolled in Tennessee's Medicaid? | eligibility | medicaid eligibility status, location based eligibility |

## What's Implemented
- All 420 subtopics across 17 topics (from the Combined sheet of Subtopic Hierarchy.xlsx)
- Controlled entity extraction for: drug names, plan names, pharmacy names, hospital/facility names, dental plan names
- Drug normalization (strips dosages/forms → base generic name only)
- Plan name fuzzy matching (against 950 official plans from plan_names.csv)
- Old grouping pipeline removed (keyword search, semantic search, similarity grouping, LLM labeling)

## Attached Files
1. **`src/data/output/questions_output.csv`** — New output (predefined subtopics) ← review this
2. **`src/data/output/ouput_question.csv`** — Old output (free-form subtopics) for comparison
3. **`src/data/input/new_sub_topic_prompt_mira.json`** — The full 420 subtopic config used

## Next Steps
- Please review the output and flag any incorrect subtopic assignments
- Would like to run on a larger batch (~100-200 real questions) for further validation
- Any adjustments to the subtopic list or entity extraction rules?

Thanks,
Vivek
