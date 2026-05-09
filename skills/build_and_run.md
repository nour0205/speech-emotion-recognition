# Skill: Build & Run

All commands assume repo root `e:/Vortrix's Projects/ISS/speech-emotion-recognition/` and a **bash** shell (Git Bash on Windows).

## Build the Unreal Plugin (Windows)
1. Right-click [unreal/EmotionDemo.uproject](../unreal/EmotionDemo.uproject) → **Generate Visual Studio project files**.
   (Or from a shell:
   `"/c/Program Files/Epic Games/UE_5.7/Engine/Build/BatchFiles/Build.bat" -projectfiles -project="<abs>/unreal/EmotionDemo.uproject" -game -rocket -progress`.)
2. Open `unreal/EmotionDemo.sln` in Visual Studio 2022.
3. Configuration: **Development Editor | Win64**, build the `EmotionDemo` target.
4. Launch the editor by double-clicking the uproject, or:
   `"/c/Program Files/Epic Games/UE_5.7/Engine/Binaries/Win64/UnrealEditor.exe" "<abs>/unreal/EmotionDemo.uproject"`.

After adding new UCLASS/USTRUCT headers, rebuild from Visual Studio (Live Coding sometimes works for .cpp-only changes; header changes almost always require a full rebuild).

## Start the Emotion API

**Local (recommended during plugin work):**
```bash
python -m venv .venv && source .venv/Scripts/activate    # Windows + Git Bash
pip install -r requirements/backend.txt
./scripts/run_api.sh --reload                            # host=0.0.0.0 port=8000
```

**Docker:**
```bash
docker compose up api
# or: docker compose up --build api
```

**Verify it's up:**
```bash
curl http://localhost:8000/health
# → {"status":"ok","model_id":"baseline","device":"cpu"}
```
Or open `http://localhost:8000/docs` (Swagger UI).

## Smoke-test the Unreal contract
```bash
curl -X POST http://localhost:8000/timeline/unreal \
  -F "file=@tests/fixtures/sample.wav" \
  -F "smoothing_method=hysteresis"
```
Expect `{ "duration_sec": <float>, "segments": [...] }`.

## Test End-to-End in the Editor
1. Start the API.
2. Open the Unreal Editor with `EmotionDemo.uproject`.
3. **Window → Emotion Bridge** (editor tab provided by `EmotionBridgeEditor`).
4. Load a WAV (file picker or Take Library).
5. Select the Amelia actor in the viewport → click **Bind Selected Actor**.
6. Click **Analyze** → wait for segments → **Play Demo**.
7. Watch Amelia's face — brow / cheeks / mouth corners should move per emotion; jaw and lips should continue being driven by the MetaHuman Post-Process ABP.

## Editor Python Helper
If Amelia's face shows zero deformation (e.g. after a MetaHuman reimport cleared the Post-Process ABP):
```python
# Output Log → switch console dropdown to Python
import importlib, sys
if 'SetupAmeliaEmotionDriver' in sys.modules:
    importlib.reload(sys.modules['SetupAmeliaEmotionDriver'])
else:
    import SetupAmeliaEmotionDriver
```
Source: [unreal/Content/Python/SetupAmeliaEmotionDriver.py](../unreal/Content/Python/SetupAmeliaEmotionDriver.py).

## Run Python Tests
```bash
pytest                       # all tests
pytest tests/api             # API slice
pytest -k timeline -q        # subset
```

## Common Errors & Fixes
- **"Cannot connect to 127.0.0.1:8000"** — API isn't running; start `./scripts/run_api.sh`.
- **"Module 'EmotionBridge' could not be loaded"** in Unreal Editor — plugin binaries are stale; rebuild from VS.
- **Amelia's face doesn't move but log shows `ApplyEmotion`** — Post-Process ABP is missing or `MetaHumanEmotionAnimInstance` isn't installed. Re-run the editor Python helper and confirm the panel shows "Face mesh bound".
- **All emotions read as neutral** — model confidence is low; check `/predict` output on the same WAV, or relax `smoothing_method`.
- **IEMOCAP returns `fear / disgust / surprise` as 0** — expected; baseline model only supports 4 labels.
- **Emotion fighting lip sync** — a preset is driving jaw/lip phoneme curves; remove them or reduce weight below 0.1 (see speech-safety rule in [skills/metahuman_curves.md](metahuman_curves.md)).
