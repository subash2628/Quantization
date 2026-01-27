# Training Data Description

## Dataset Overview

**File:** `contrastive_rec_train.jsonl`  
**Format:** JSON Lines (JSONL) - one JSON object per line  
**Total Examples:** 6,040  
**Task Type:** Movie Recommendation with Contrastive Learning  

## Dataset Purpose

This dataset trains a language model to:
1. Analyze a user's movie viewing history
2. Identify genre and theme preferences
3. Compare two candidate movies (Option A vs Option B)
4. Recommend the better match with clear reasoning

## Data Structure

Each example contains three fields:

### 1. **instruction** (constant across all examples)
```
"Analyze the user's history to identify their preference. Compare two potential movies and explain which one is the better recommendation."
```

### 2. **input** (variable)
Contains:
- **History:** 5 movies the user has watched (with year)
- **Option A:** First candidate movie with genres
- **Option B:** Second candidate movie with genres

**Example:**
```
History: Toy Story (1995), Lion King, The (1994), Beauty and the Beast (1991), 
Aladdin (1992), Mulan (1998).
Option A: Pocahontas (1995) (Animation|Children's|Musical|Romance).
Option B: Turbo: A Power Rangers Movie (1997) (Action|Adventure|Children's).
```

### 3. **output** (target response)
A structured recommendation following this format:
```
The better recommendation is Option [A/B]. Reasoning: The user has shown a strong 
preference for themes found in [genres]. Option [other] ([genres]) does not align 
with their recent viewing patterns.
```

## Genre Distribution

The dataset covers **18 unique movie genres:**

1. Action
2. Adventure
3. Animation
4. Children's
5. Comedy
6. Crime
7. Documentary
8. Drama
9. Fantasy
10. Film-Noir
11. Horror
12. Musical
13. Mystery
14. Romance
15. Sci-Fi
16. Thriller
17. War
18. Western

## Dataset Characteristics

### Contrastive Learning Structure
- **Positive Examples Only:** All 6,040 examples recommend Option A
- **Option A:** Always matches the user's preference pattern
- **Option B:** Always represents a contrasting/mismatched choice
- This creates a contrastive learning scenario where the model learns to distinguish good matches from poor matches

### Example Patterns

**Pattern 1: Animation Preference**
- History: Animated Disney films
- Option A: Similar animation (✓ recommended)
- Option B: Action/Adventure (✗ not recommended)

**Pattern 2: Action/Thriller Preference**
- History: Action-heavy films
- Option A: Similar action/thriller (✓ recommended)
- Option B: Drama or children's film (✗ not recommended)

**Pattern 3: Classic Film Preference**
- History: Critically acclaimed classics
- Option A: Another classic/acclaimed film (✓ recommended)
- Option B: Contemporary genre film (✗ not recommended)

## Data Quality Notes

### Strengths
- Consistent format across all examples
- Clear reasoning structure
- Covers diverse genres
- Based on real movie viewing patterns (likely from MovieLens dataset)

### Potential Limitations
- **Imbalanced:** 100% Option A wins (no Option B wins)
- This may cause the model to:
  - Develop bias toward Option A
  - Struggle with cases where Option B is actually better
  - Not learn true preference matching, but rather pattern matching to always choose A
- **Limited diversity** in reasoning templates (same structure repeated)

## Recommended Training Considerations

1. **Balance the dataset:** Add examples where Option B wins
2. **Increase epochs:** Current training uses only 1 epoch (755 steps)
3. **Validate carefully:** Test with balanced test cases where either option could be correct
4. **Monitor for bias:** Check if model always prefers Option A regardless of input

## Usage in Training

This dataset is formatted for **Supervised Fine-Tuning (SFT)** with the following template:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

The model learns to generate the response given the instruction and input context.

## Source Dataset

Based on file names in the workspace (`movielens_train.jsonl`), this appears to be derived from the **MovieLens** dataset, which contains real user movie ratings and viewing histories.
