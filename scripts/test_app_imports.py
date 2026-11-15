import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

def p(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

p("Importing json...")
import json
p("Importing pathlib...")
from pathlib import Path
p("Importing typing...")
from typing import Any, Dict, List, Optional
p("Importing re...")
import re
p("Importing fastapi...")
from fastapi import FastAPI, HTTPException, Path as FastAPIPath, Query, Response, status
p("Importing CORSMiddleware...")
from fastapi.middleware.cors import CORSMiddleware
p("Importing FileResponse...")
from fastapi.responses import FileResponse
p("Importing StaticFiles...")
from fastapi.staticfiles import StaticFiles
p("Importing pydantic...")
from pydantic import BaseModel, Field
p("Importing get_data_loader...")
from src.web.data_loader import get_data_loader
p("Importing math...")
import math
p("Now importing src.web.app directly...")
import src.web.app
p("Imported src.web.app successfully!")

