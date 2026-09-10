## 🛠️ Proposed Solution (by Aditya Waghamare)

### Analysis
The bug arises because `tokenizer_fallback` was hardcoded or logically evaluated with `tokenizer == "approx"`, marking standard approximate counts as fallbacks.

### Fix
Modify `src/tirzah/retrieval/budget.py`:
