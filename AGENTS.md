# Agent Instructions

## Project Structure

```
amazon_review_sentiment_analysis/
  phase1/
    data/                    # raw dataset files (not committed)
    01_data_exploration.py   # dataset EDA: counts, distributions, duplicates, plots
    02_preprocessing.py      # labeling, outlier removal, text cleaning, 1000-review sample
    03_lexicon_modeling.py   # VADER and TextBlob sentiment prediction
    04_evaluation.py         # metrics (accuracy, precision, recall, F1, confusion matrix) + comparison table
```


## Code Style and Complexity

### Variable and File Naming
- Name files and variables using snakecase
- Name constants using caps snakecase

### Avoid Over-Engineering
- Do not create unnecessary function wrappers
- Do not add features that weren't requested
- Do not use verbose output formats:
  - No caps-locked headers with `=*80` dividers (e.g., "=== HEADER ===")
  - No markdown-style headers with equals (e.g., "==Header==")
  - If separation is needed, use dashes followed by header with approximately 80-100 dashes for full screen width:
    - Format: `"-" * 80 + "\n" + "Header"`
    - Example: `"--------------------------------------------------------------------------------\nSection Title"`
- Keep code at the appropriate level for the assignment context

### Comments
- **Do not add author or date comments** unless explicitly asked to
- Write comments above major code blocks or complex lines
- Keep comments concise and relevant

## Package Management

### UV
- This project uses **UV** as the package manager
- **Never use pip, pip install, or any pip commands**
- **Never use conda or other package managers**
- To add dependencies, use: `uv add <package>`
- To run Python scripts, use: `.venv/Scripts/python.exe <script>` or the appropriate venv path
- Do not attempt to modify package management - UV handles everything

## Report (`report.md`)

The `report.md` serves as the project report draft for code-related content only. Do **not** write cover pages, table of contents, project plans, meeting logs, or peer-evaluation sections — those are handled outside of code.

### Report Guidelines
- Follow `project_spec.txt` for the full list of required deliverables and report requirements (sections 1–10 of Phase 1, project report requirements).
- Every decision must be justified (column selection, outlier strategy, lexicon choice, pre-processing steps).
- All results must reference the script that produced them.
- Do not duplicate content already captured as code comments.
- Do not add project-management content (plans, meeting logs, peer evaluation).

## File Operations

### Be Cautious with File Generation
- **Do not run code that downloads or creates multiple files** without explicit permission
- Always review what files will be created before running code
- Ask before running scripts that perform file I/O operations if uncertain

### File Paths for I/O
- Use the location of the current file when working with I/O operations
- **Do not use `os.getcwd()`** as it causes issues depending on how the file is run
- Use `os.path.dirname(os.path.abspath(__file__))` to get the script's directory

