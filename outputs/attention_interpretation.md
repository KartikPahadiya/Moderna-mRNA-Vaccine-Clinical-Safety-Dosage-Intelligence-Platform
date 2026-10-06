# Self-attention interpretation (15 bases)  
--------------------------------------  
- Every row sums to 1.0, so each base redistributes a fixed budget of  
  "attention" across all positions — no information is created or lost.  
- Position 1 (C) attends most strongly to itself  
  (weight 0.07); identical bases share the same embedding,  
  so same-base pairs score high dot-product similarity in Q·Kᵀ.  
- Position 2 shows the flattest distribution (std 0.001),  
  spreading attention nearly evenly — it is "unsure" what matters.  
- Because W_q/W_k/W_v are randomly initialized, patterns here reflect  
  embedding geometry, not learned biology; training would sharpen them.  
