---
description: "Refactor Python code for clarity, consistency, and cross-file correctness."
---

Refactor the following Python code while fully preserving its external behaviour.

Requirements:
- Improve **readability**, **structure**, and **maintainability**.  
- Make functions or variables names concise to enhance clarity.
- Apply **consistent naming**, **type hints**, and short **Google-style docstrings**.  
- Remove **duplication**, **deep nesting**, and **unused variables/imports**.  
- Update or add **function/method-level comments** at the beginning, summarising purpose and behaviour.  
- If this refactor affects related functions, classes, or modules in other files (e.g. renamed symbols, updated interfaces, changed parameters), **update all those references consistently and correctly**.  
- Ensure the refactored code integrates seamlessly with the rest of the project (imports, calls, tests).  
- Avoid too many small functions; balance modularity with readability.  
- Do not change any logic or functionality.
- Preserve unit-test compatibility.

```python
${selection}
