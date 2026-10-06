# Retrieval log — Module 3

| # | Question | Top retrieved chunk (truncated) | Latency (s) |
|---|---|---|---|
| 1 | What are the storage requirements for unopened mRNA formulations? | Unopened mRNA lipid nanoparticle formulations are stored at ultra-low temperature, typically between -25C and -15C for e... | 0.037 |
| 2 | What dose is recommended for adults aged 65 and older with two pre-existing conditions? | Ages 65 and older: standard adult dosing range applies; however, patients in this age band with two or more relevant pre... | 0.02 |
| 3 | When should a dose be deferred because of an elevated inflammatory biomarker? | Patients presenting with an inflammatory biomarker level meaningfully above the normal reference range at the time of sc... | 0.018 |
| 4 | What are the absolute contraindications for administering the vaccine? | 3.1 Absolute Contraindications  Known severe allergic reaction (e.g., anaphylaxis) to a prior dose of the same product o... | 0.021 |
| 5 | When must an adverse event be escalated to clinical safety review? | 4.2 Escalation Criteria  An adverse event should be escalated to clinical safety review if it involves hospitalization, ... | 0.019 |

Chunking choice: RecursiveCharacterTextSplitter with chunk_size=500 characters and 50-character overlap. The source document is only ~1,200 words, so this yields 17 chunks of 16-70 words — small enough that each rule stays whole and retrieval stays specific. A literal 300-500 *token* chunk size would have produced just 2-3 chunks and destroyed retrieval granularity, so the character budget was tuned to the document length instead (documented trade-off vs FR-3.1).

Known failure case: questions mixing two topics (e.g. storage AND dosing) sometimes return a chunk covering only one of them, because a single chunk cannot span two document sections. Hypothesis: increasing overlap would not fix it; cross-encoder re-ranking of more candidates (k=10 -> top 3) would.