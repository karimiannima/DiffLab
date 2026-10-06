# Publishing to GitHub

```bash
# 1. Create empty repo on GitHub: karimiannima/DiffLab (no README)

# 2. From this folder:
git remote add origin git@github.com:karimiannima/DiffLab.git   # skip if exists
git branch -M main
git push -u origin main

# 3. GitHub → Settings → Pages → Deploy from branch `main` / root
# Live demo: https://karimiannima.github.io/DiffLab/
```

Or: `./scripts/push.sh`
