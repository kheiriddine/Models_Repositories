#!/bin/bash
apt-get update && apt-get install -y python3-distutils
git clone --filter=blob:none --no-checkout https://github.com/kheiriddine/Models_Repositories/
cd Models_Repositories
git sparse-checkout init --cone
git sparse-checkout set Models_Repositories  # Exclude /models/
git checkout
