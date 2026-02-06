## Setup & Installation

This project uses **[uv](https://github.com/astral-sh/uv)** for dependency management.

### 1. Install uv
If you don't have `uv` installed, get it with one command:

**macOS / Linux:**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

**windows:**
```powershell
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

### 2. Install dependencies
Run this command in the project root. It will create the virtual environment and install the exact versions locked in `uv.lock`.

```bash
uv sync
```

If this doesn't work please contact me.

### 3. Download the Model
This project requires a custom-trained model. You can download it automatically by running:

```bash
uv run setup.py
```

### 4. Running the code
Use `uv run` to execute scripts (this automatically uses the virtual environment):
```bash
uv run test.py
```
