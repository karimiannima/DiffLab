# Fix / enable GitHub Pages for DiffLab

Live URL (once published): **https://karimiannima.github.io/DiffLab/**

## Recommended: GitHub Actions (this repo includes the workflow)

1. Open https://github.com/karimiannima/DiffLab/settings/pages  
2. Under **Build and deployment → Source**, choose **GitHub Actions**  
3. Open the **Actions** tab → allow workflows if prompted  
4. Re-run workflow **Deploy DiffLab to GitHub Pages** (or push any commit to `main`)  
5. Wait ~1–2 minutes → open https://karimiannima.github.io/DiffLab/

## Alternative: Deploy from branch

1. Same Pages settings page  
2. Source: **Deploy from a branch**  
3. Branch: **`main`** (or **`gh-pages`**) / folder: **`/ (root)`**  
4. Save → wait for green check on the Pages settings page  

## If you still get 404

| Check | Fix |
|--------|-----|
| Repo is **private** on a free plan | Set repo to **Public** (Settings → General → Danger zone), *or* use a plan that includes private Pages |
| Source never set | Set Source as above (Pages does nothing until this is saved once) |
| Workflow failed | Actions tab → open failed run → “Deploy to GitHub Pages” needs `pages: write` (workflow already sets this) |
| Wrong URL | Must be `https://karimiannima.github.io/DiffLab/` (trailing path = repo name) |
| Browser cache | Hard refresh or incognito |

## Local sanity check

```bash
python -m http.server 8080
# http://localhost:8080
```
