# GitHub Pages for DiffLab

## Demo that works immediately (no Pages config)

Open either:

- **https://raw.githack.com/karimiannima/DiffLab/main/index.html**
- **https://cdn.jsdelivr.net/gh/karimiannima/DiffLab@main/index.html**

These serve the same files from `main` and load ES modules correctly.

## Official URL (after Pages is configured)

https://karimiannima.github.io/DiffLab/

### Enable it

1. https://github.com/karimiannima/DiffLab/settings/pages  
2. **Build and deployment → Source**  
   - **Deploy from a branch** → branch **`main`** → folder **`/ (root)`** → Save  
   - *or* **GitHub Actions** → Actions tab → run **Deploy DiffLab to GitHub Pages**
3. Wait until the settings page shows: *Your site is live at https://karimiannima.github.io/DiffLab/*

### Still 404 on github.io?

- Confirm the green “live” message on the Pages settings page (not only that the repo is public).
- Check **Actions** for a failed deploy workflow.
- Try `gh-pages` branch as source instead of `main`.
- Hard refresh / incognito.
- Propagation can take a few minutes after the first successful deploy.

Repo flags we expect: **Public**, `index.html` + `.nojekyll` at branch root.
