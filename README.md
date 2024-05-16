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

To upload to S3, require amazon AWS CLI to be installed. 

```
doc2html pdf_or_latex_file_or_folder --s3
```