```markdown
<theory>
ここに貼り付け
</theory>

<guideline>
**Principle:**
Be extremely concise and sacrifice grammar for the sake of concision.

**Procedure:**
1. Keep all formulas/equations intact to ensure theoretical validity and traceability.
2. Aggressively remove all explanations *except* the absolute minimum required.
</guideline>

<task>
Refine the theoretical description
</task>
```

```markdown
<code>
</code>
<guideline>
**Procedure:**
1. All symbols to be more short semantic name.
2. Remove all unnecessary code.
</guideline>
<task>
Refine the code
</task>
```

```markdown
# Roles
<YOU> Professor & Prompt Revisor: An orchestrator who gives instructions to specialized LLMs of answer USER questions when unsure of the policy. </YOU>
Developer: LLM who have code access with dev dependencies installed.
DeepResearcher: Specialized LLM for research (test-time diffution model).
DeepThink: 

# Context

## Overview
自動車の車体部分の熱のこもりやすさを概算できる理論が欲しい.
実験により測定するのがとても大変だからである.
車体は、クリアコート層、塗料の層、金属板、の三層構造で単純化されるとする.
熱の原因として、800 ~ 2500のNIRに対する反射吸収スペクトルを考える.

## Code to demonstrate clear coat

# My Concerns
My thinking is that for the clear coat, the interference fringes at a 20µm thickness are extremely fine. When you consider the integral of the power across the entire 800-2500 nm range, the net effect of the fringes seems like it would average out to a negligible amount.
However, the real problem is the paint layer underneath. The paint particles are large and act as obstacles, meaning Mie scattering and the Kubelka-Munk theory are more relevant. This implies that not only the phase but also the angle of the scattered light will become randomized. This is a problem because the Transfer Matrix Method (TMM) requires a specific angle as an input.
This makes me think that perhaps I should give up on using TMM altogether.
Ultimately, my question is: Isn't there an established method for calculating the optics of a domain like a clear coat on top of a paint layer? I need to research this. Searching in an automotive context didn't yield results, so I feel like I need to search in a more theoretical context.

# Task
Build prompt for DeepResearcher in English.
```
