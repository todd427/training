# enver.py
# print the environment variables for the current python environment
# useful for debugging
import json, platform, torch, sys, subprocess
def ver(pkg):
    try:
        m=__import__(pkg); return getattr(m,"__version__", "unknown")
    except Exception:
        return None
env={
 "python": sys.version,
 "platform": platform.platform(),
 "torch": getattr(torch,"__version__",None),
 "cuda": getattr(torch.version,"cuda",None),
 "transformers": ver("transformers"),
 "accelerate": ver("accelerate"),
 "peft": ver("peft"),
 "tokenizers": ver("tokenizers"),
 "datasets": ver("datasets"),
 "bitsandbytes": ver("bitsandbytes"),
}
print(json.dumps(env, indent=2))