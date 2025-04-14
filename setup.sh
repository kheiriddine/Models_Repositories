#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y python3-distutils git-lfs

# Clone without LFS files
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kheiriddine/Models_Repositories /mount/src/repo


