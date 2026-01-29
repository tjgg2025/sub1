# Diversity of Thoughts

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```

3. (Optional) For Together AI models, modify the API key in `generators/model.py` line 62:
```python
api_key=os.environ.get("TOGETHER_API_KEY", "xxxxx")
```
Or set the environment variable:
```bash
export TOGETHER_API_KEY=<your key>
```

## Usage

See `run_code.sh` for example commands.
