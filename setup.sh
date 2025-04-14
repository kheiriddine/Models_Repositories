#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y python3-distutils git-lfs

# Partial clone (exclude large files)
git clone --filter=blob:none --no-checkout https://github.com/kheiriddine/Models_Repositories /mount/src/repo
cd /mount/src/repo

# Sparse checkout (only needed files)
git sparse-checkout init --cone
git sparse-checkout set dashboard_git.py requirements.txt  # Explicitly list required files

# Optional: Manual LFS pull for specific small files if absolutely needed
# git lfs pull --include="path/to/small_model.bin" --exclude="*"
