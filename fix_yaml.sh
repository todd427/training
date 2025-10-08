V3=~/training/ckpts/toddric-3b-merged-v3/README.md
V3B=~/training/ckpts/toddric-3b-merged-v3-bnb4/README.md


export HF_HUB_ENABLE_HF_TRANSFER=1

hf upload toddie314/toddric-3b-merged-v3      "$(dirname "$V3")"  --repo-type model --revision main --commit-message "Add YAML metadata"
hf upload toddie314/toddric-3b-merged-v3-bnb4 "$(dirname "$V3B")" --repo-type model --revision main --commit-message "Add YAML metadata"

