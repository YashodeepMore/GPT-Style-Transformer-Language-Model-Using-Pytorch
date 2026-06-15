# TinyStories Full-Stack Generator

A local React and FastAPI application powered by the custom Transformer
architecture in `training_model_refrence.py`.

## Checkpoint placement

The model checkpoint is not committed in this workspace. Place
`tinystories_model.pt` in:

```text
backend/tinystories_model.pt
```

The standalone inference script also accepts a checkpoint at the project root.

## 1. Verify inference

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python test_inference.py
```

The script automatically selects CUDA when available, loads the checkpoint
vocabulary and configuration, strictly loads the model state dict, and
generates from `Once upon a time`.

## 2. Run the backend

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```

The API runs at `http://localhost:8000`. Interactive docs are available at
`http://localhost:8000/docs`.

## 3. Run the frontend

Open another PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

To use a different API address, create `frontend/.env`:

```text
VITE_API_URL=http://localhost:8000
```

## API

`POST /generate`

```json
{
  "prompt": "Once upon a time"
}
```

The response contains up to four non-empty stories split on `**EOS**`.
