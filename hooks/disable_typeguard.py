# hooks/disable_typeguard.py
import os
os.environ["TYPEGUARD_DISABLE_RUNTIME_TYPE_CHECKING"] = "1"
