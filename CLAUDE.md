# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains work for the Multi-Objective Optimization course (EEE910, 2025/02) at UFMG. The course covers multi-objective optimization theory, algorithms, Pareto optimality, quality indicators, and multi-attribute decision-making methodologies.

## Project Structure

- `TC1/` - Computational Work 1: Task Assignment Problem with bi-objective optimization
  - `data_5x50_*.csv` - Problem instance data (5 agents, 50 tasks)
  - `data_5x50.mat` - Same data in MATLAB format
  - `TC.pdf` - Complete problem specification
  - `apresentações/` - Presentation deliverables

## TC1: Task Assignment Problem

### Problem Description

A company has `n` tasks to assign to `m` agents. The problem uses a 5-agent, 50-task instance.

**Input Data Files:**
- `data_5x50_m.csv` - Number of agents (m)
- `data_5x50_n.csv` - Number of tasks (n)
- `data_5x50_a.csv` - Resource requirements matrix `a[i,j]` (resources needed by agent i for task j)
- `data_5x50_c.csv` - Cost matrix `c[i,j]` (cost of assigning task j to agent i)
- `data_5x50_b.csv` - Capacity vector `b[i]` (total resource capacity of agent i)

### Objective Functions

1. **f_C(·)** - Minimize total cost of completing all tasks
2. **f_E(·)** - Minimize workload imbalance (difference between most and least occupied agent)

These objectives are conflicting and form a bi-objective optimization problem.

### Constraints

1. Agent capacity cannot be violated: Σ_j (a[i,j] × x[i,j]) ≤ b[i] for all agents i
2. Each task assigned to exactly one agent: Σ_i x[i,j] = 1 for all tasks j
3. Binary decision variables: x[i,j] ∈ {0,1}

## Required Analysis Workflow

### 1. Mathematical Modeling (Entrega #1)
- Define all parameters, variables, objective functions, and constraints explicitly
- Ensure objectives are conflicting

### 2. Algorithm Selection (Entrega #2)
- Choose and justify optimization algorithm(s) for the bi-objective problem
- Common approaches: scalarization methods, NSGA-II, MOEA/D, weighted sum, ε-constraint

### 3. Implementation and Results (Entrega #3)

**For non-exact methods:**
- Run algorithm 5 times
- Overlay all 5 Pareto fronts in a single plot

**Quality Indicators:**
- Calculate Delta (Δ) and Hypervolume at 10 equally-spaced intervals during each run
- Delta: Use utopian point (or approximation)
- Hypervolume: Use reference vector = 1.1 × anti-utopian vector
- Plot average evolution curves for both indicators across runs

**Decision Making:**
- Merge all Pareto fronts if needed, extract non-dominated solutions
- Select max 20 uniformly distributed solutions from final front
- Define 4+ decision attributes: the 2 objectives + at least 2 additional attributes (e.g., robustness, reliability, environmental impact)
- **All attributes must be conflicting**
- Use AHP to determine attribute weights
- Apply decision methods: AHP, ELECTRE, PROMETHEE, TOPSIS
- Plot final front with chosen solution(s) highlighted
- Visualize final solution characteristics

## Key Requirements

- All computational work must include formal article using provided templates
- Submit both article and code via Moodle
- Groups: maximum 10 groups, smallest size possible
- Deliverables include presentations (5-15 min) + article + code
- Students absent from presentations receive no points

## Decision-Making Methods

When implementing decision-making analysis:
- **AHP** (Analytic Hierarchy Process): Define attribute weights through pairwise comparisons
- **ELECTRE**: Outranking method based on concordance/discordance
- **PROMETHEE**: Preference ranking based on pairwise comparisons
- **TOPSIS**: Technique for Order Preference by Similarity to Ideal Solution

Handle incomparability between alternatives by establishing additional criteria when needed.

## Course Schedule Context

- Week 7 (09/10): TC1a presentation (mathematical modeling - 5 min)
- Week 10 (30/10): TC1b presentation (optimization tool - 5 min)
- Week 14 (04/12): TC1c presentation (final results + article + code - 15 min)
