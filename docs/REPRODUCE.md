# Step-by-step reproduction

```bash
pip install -r requirements.txt
python verify.py xval 13.9   # bitwise cross-check against the official bsm2-python
python sweep.py all           # 261-point steady-state scan (~12 min on 16 cores)
python make_figures.py
```

The archived outputs in `data/` are exactly the files produced by these commands.
