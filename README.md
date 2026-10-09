# Student Performance & Learning-Risk Analytics

This project keeps the original Streamlit application in [app.py](./app.py) and
adds a Vercel-compatible web adapter. The Vercel version uses a Next.js
frontend and Python serverless functions for dataset loading and inference.
The existing `models/risk_model.pkl` and datasets are used as-is.

## Architecture and Vercel decision

Streamlit is a persistent Python server and cannot be deployed directly to
Vercel by adding a `vercel.json` file. The adapter is the smallest reversible
conversion:

- `app/` provides the five dashboard pages and CSV upload UI.
- `api/data.py` loads the bundled dataset.
- `api/predict.py` loads the existing Joblib pipeline and performs predictions.
- `requirements.txt` contains only the Python packages required by the API.
- `app.py` remains available for the original Streamlit workflow.

The model receives only `G1`, `G2`, `studytime`, `absences`, `failures`, and
`age`, in that order. `G3` is never sent to the model.

## Local setup

### Original Streamlit application

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-streamlit.txt
streamlit run app.py
```

If no virtual environment exists, create one with `py -m venv .venv` first.

### Vercel-compatible application

Install Node.js 18.17+ and Python 3.10+.

```powershell
npm install
pip install -r requirements.txt
npm run typecheck
npm run build
npx vercel dev
```

Open `http://localhost:3000`. The Python functions are served by Vercel's
development runtime when using `vercel dev`; `next dev` alone serves the
frontend but does not emulate Python functions.

## Deploy to Vercel

1. Push this repository to GitHub.
2. In Vercel, choose **Add New > Project**, then import
   `BharathChandra4545/mini-project-1`.
3. Set the **Root Directory** to the repository root (the directory containing
   `package.json`, `app/`, `api/`, `models/`, and `data/`).
4. Keep the detected framework as **Next.js**.
5. Use `npm install` as the install command and `npm run build` as the build
   command. Leave the output directory at the Next.js default.
6. Do not add environment variables: this project does not require credentials
   or private configuration.
7. Click **Deploy**, then test the deployed URL using the checklist below.

Vercel detects `api/*.py` as Python Functions and installs the root
`requirements.txt`. The model and CSV files must remain in the repository so
the functions can load them at runtime.

## Limitations

- Vercel Python Functions are stateless and have execution/package-size limits.
  Large CSV uploads or very large datasets may need object storage or a
  dedicated Python host later; no database is required for this project.
- The frontend parses uploaded CSV text in the browser and sends rows to the
  inference function. The bundled dataset is loaded by the Python function.
- The serialized model should be rebuilt only when intentionally changing the
  training environment. It is not replaced by this deployment adapter.

## Deployment test checklist

- Open each of the five navigation pages.
- Confirm the overview metrics and grade distribution load.
- Upload both a comma-delimited and semicolon-delimited student CSV.
- Submit the prediction form and verify a probability and classification are
  returned from the trained model.
- Generate the student report and download the CSV.
- Try a CSV missing one of `G1`, `G2`, `studytime`, `absences`, `failures`, or
  `age`; confirm a readable validation error appears.
- Confirm the browser network panel shows successful `/api/data` and
  `/api/predict` requests.
