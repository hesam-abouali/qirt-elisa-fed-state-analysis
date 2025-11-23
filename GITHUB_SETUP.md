# GitHub Repository Setup Instructions

Follow these steps to upload your code to GitHub.

## Step 1: Create Repository on GitHub Website

1. Go to https://github.com
2. Click the **"+"** button in top-right corner
3. Select **"New repository"**
4. Fill in:
   - Repository name: `qirt-elisa-fed-state-analysis`
   - Description: "Analysis pipeline for QIRT-ELISA fed-state metabolic phenotyping (NeurIPS 2025)"
   - Choose: **Public** (so others can access it)
   - **DO NOT** check "Initialize with README" (we already have one)
5. Click **"Create repository"**

## Step 2: Initialize Git Repository Locally

Open terminal and navigate to your repository folder:

```bash
cd /path/to/github_repo
```

Initialize git:

```bash
git init
```

## Step 3: Add All Files

```bash
git add .
```

## Step 4: Commit Changes

```bash
git commit -m "Initial commit: QIRT-ELISA fed-state analysis pipeline

- Added comprehensive feature extraction pipeline (184 metrics)
- Included de-identified experimental data (n=9 animals)
- Added documentation and usage examples
- Included MIT license
- Supporting NeurIPS 2025 TS4H workshop paper"
```

## Step 5: Connect to GitHub

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/qirt-elisa-fed-state-analysis.git
```

## Step 6: Push to GitHub

```bash
git branch -M main
git push -u origin main
```

You may be prompted to log in to GitHub. Use your credentials.

## Step 7: Verify Upload

Go to: https://github.com/YOUR_USERNAME/qirt-elisa-fed-state-analysis

You should see all your files!

---

## Optional: Add Topics (Tags)

On your GitHub repository page:

1. Click ⚙️ (Settings icon) next to "About"
2. Add topics:
   - `machine-learning`
   - `diabetes`
   - `time-series`
   - `biomedical-engineering`
   - `neurips2025`
   - `healthcare`
   - `qirt-elisa`
   - `python`

---

## Troubleshooting

### If you get authentication errors:

**Option A: Use Personal Access Token (Recommended)**

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` permissions
3. Use the token as your password when pushing

**Option B: Use SSH (More Secure)**

1. Generate SSH key:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
2. Add SSH key to GitHub: Settings → SSH and GPG keys
3. Change remote URL:
   ```bash
   git remote set-url origin git@github.com:YOUR_USERNAME/qirt-elisa-fed-state-analysis.git
   ```

### If you need to make changes later:

```bash
# Make your changes, then:
git add .
git commit -m "Description of changes"
git push
```

---

## Next Steps

After uploading to GitHub:

1. ✅ Update your NeurIPS paper with the GitHub URL
2. ✅ Add repository link to your paper's "Code Availability" section
3. ✅ Share the link with collaborators
4. ✅ Tweet/share your work!

---

**Your GitHub Repository URL will be:**
```
https://github.com/YOUR_USERNAME/qirt-elisa-fed-state-analysis
```

Use this link in your paper!
