## Prerequisites

```python
pip install -e .
```

For .eps image conversion to install ghostscript. For Mac OS

```
brew install ghostscript  
```

### Conversion

```
doc2html pdf_or_latex_file_or_folder
```

If the target is a folder, it will recursively search for any main.tex or main.pdf files. 

To run with alt text accessibility

```
doc2html pdf_or_latex_file_or_folder --accessibility
```

To upload to S3, require amazon AWS CLI to be installed. 

```
doc2html pdf_or_latex_file_or_folder --s3
```

To run with a model other than the default (GPT-4o), e.g.

```
doc2html pdf_or_latex_file_or_folder --model gpt-4 --vision_model gpt-4-vision-preview
```