## Prerequisites

```python
pip install -U jupyter-book
pip install openai tiktoken sphinx-exercise sphinx-proof 
```

For .eps image conversion to install ghostscript. For Mac OS

```
brew install ghostscript  
```


## OpenAI conversion

From preliminary investigation, `gpt-4` works best, suffering fewer mistakes and hallucinations. However, it costs 3x 
more than GPT 4 Turbo. It which can be invoked by  
