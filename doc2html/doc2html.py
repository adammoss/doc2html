"""
Converts latex files to markdown using OpenAI
"""

from openai import OpenAI
import tiktoken

import fitz
import argparse
import os
import re
import time
from pathlib import Path
import base64
import requests
import json
import PIL.Image
import io
import urllib.request
import shutil
from contextlib import contextmanager
import yaml
import tempfile
import numpy as np
import glob


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_accessibility(image_path, api_key, vision_model='gpt-4-vision-preview'):
    try:
        base64_image = encode_image(image_path)
    except FileNotFoundError:
        print('Could not find file: %s' % image_path)
        return 'Could not generate alt text', 0

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Give alt text for this image in 100 words or less. DO NOT use : in the text."
                                "At the end, give a minimum width for the image in pixels to be readable on a webpage, "
                                "in the format <..px>"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300,
    }

    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    if 'choices' not in resp:
        print(resp)
        return 'Could not generate alt text', 0
    message = resp['choices'][0]['message']
    recommended_width = re.search('.*?<(\d+)px>.*', message['content'])
    alt_text = re.sub('<(\d+)px>', '', message['content']).replace(':', '-')
    result = alt_text
    if recommended_width is not None:
        result += '\nIt should have a width %spx' % recommended_width.group(1)
    return result, resp['usage']['total_tokens']


signature_get_accessibility = {
    "name": "get_accessibility",
    "description": "Returns alt text of an image and recommended image width. "
                   "Always call this function whenever there is a figure. ",
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "The path of the image.",
            },
        },
        "required": ["image_path"],
    }
}


def get_latex_diagram(latex_string, filename):
    doc = "\\documentclass[10pt]{article}\n" \
          "\\usepackage[usenames]{color}\n" \
          "\\usepackage{amssymb}\n" \
          "\\usepackage[utf8]{inputenc}\n" \
          "\\usepackage{feynmp-auto}\n" \
          "\\begin{document}\n\pagenumbering{gobble}\n" \
          "%s\n" \
          "\\end{document}" % latex_string
    with open(os.path.join(tempfile.gettempdir(), 'tmp.tex'), 'w') as f:
        f.write(doc)
    with cd(tempfile.gettempdir()):
        os.system('pdflatex tmp.tex')
        mpost_files = glob.glob(os.path.join(tempfile.gettempdir(), '*.mp'))
        for f in mpost_files:
            os.system('mpost %s' % f)
        os.system('pdflatex tmp.tex')
    doc = fitz.open(os.path.join(tempfile.gettempdir(), 'tmp.pdf'))
    doc[0].get_pixmap(dpi=200).save(os.path.join(tempfile.gettempdir(), 'tmp.png'))
    image = PIL.Image.open(os.path.join(tempfile.gettempdir(), 'tmp.png'))
    image_data = np.asarray(image)
    image_data_bw = image_data.max(axis=2)
    non_empty_columns = np.where(image_data_bw.min(axis=0) < 10)[0]
    non_empty_rows = np.where(image_data_bw.min(axis=1) < 10)[0]
    crop_box = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    image_data_new = image_data[crop_box[0]-10:crop_box[1] + 10 + 1, crop_box[2] - 10:crop_box[3] + 10 + 1, :]
    new_image = PIL.Image.fromarray(image_data_new)
    if '.png' not in filename:
        filename += '.png'
    new_image.save(os.path.join(tempfile.gettempdir(), filename))
    return os.path.join(tempfile.gettempdir(), filename)


signature_latex = {
    "name": "get_latex_diagram",
    "description": "Generates a diagram from latex code.",
    "parameters": {
        "type": "object",
        "properties": {
            "latex_string": {
                "type": "string",
                "description": 'The latex code used to generate the diagram.'
            },
            "filename": {
                "type": "string",
                "description": 'The output filename. Ensure it is unique.'
            },

        },
        "required": ["latex_string", "filename"],
    }
}


def transcribe_image(image_path, api_key, prompt, vision_model='gpt-4-vision-preview'):
    try:
        base64_image = encode_image(image_path)
    except FileNotFoundError:
        return 'Could not find image'

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    message = resp['choices'][0]['message']
    return message['content'], resp['usage']['total_tokens']


def get_figure_coordinates(image_path, api_key, vision_model='gpt-4-vision-preview'):
    try:
        base64_image = encode_image(image_path)
    except FileNotFoundError:
        return 'Could not find image'

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Can you tell me the location of the figure in the image. "
                                "Share the x_min, y_min, x_max and y_max in 0-1 normalized space. "
                                "Only return the numbers, nothing else. "
                                "Return them in the format (x_min, y_min, x_max, y_max)"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000,
    }

    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    message = resp['choices'][0]['message']
    return message['content'], resp['usage']['total_tokens']


def execute_function_call(function_name, fn_args, args, api_key):
    fn_tokens_used = 0
    if function_name == "get_accessibility":
        fn_args["api_key"] = api_key
        fn_args["image_path"] = os.path.join(args.out, "Chapters", fn_args["image_path"])
        fn_args["vision_model"] = args.vision_model
        if '.pdf' in fn_args["image_path"]:
            fn_args["image_path"] = fn_args["image_path"].replace('.pdf', '.png')
        if '.eps' in fn_args["image_path"]:
            fn_args["image_path"] = fn_args["image_path"].replace('.eps', '.png')
        content, fn_tokens_used = get_accessibility(**fn_args)
    if function_name == "get_latex_diagram":
        filename = get_latex_diagram(**fn_args)
        shutil.copy(filename, os.path.join(args.out, "Chapters"))
        content = "The filename %s has been produced and can be inserted into an " \
                  "figure directive" % (fn_args['filename'] + '.png')
    else:
        content = f"Error: function {function_name} does not exist"
    return content, fn_tokens_used


def num_tokens(chunk, model="gpt-4-1106-preview"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(chunk)))


def get_system_prompt(mode="tex", accessibility=True, figure_paths=None):

    if accessibility:
        image_comment = 'ALWAYS use the function call to get image accessibility information (alt text and width)' \
                        ', whenever there is a figure or graphics. Do NOT use the caption as alt text. '
        alt_text_comment = 'ONLY insert the returned alt text from the available function call. '
        width_comment = 'Use the recommended width from the available function call, otherwise set to 500px. '
    else:
        image_comment = ''
        alt_text_comment = 'alt text'
        width_comment = '500px'
    if figure_paths is not None:
        image_comment += 'The figure filenames to use in the order figures appear on the page ' \
                         'are ' + ', '.join(figure_paths) + ' .'

    myst_prompt = 'Use MyST ONLY in the following cases: ' \
                  '(1) Definitions in the format ' \
                  '"````{admonition} admonition-title \n:class: .... \n .... \n````". ' \
                  '(2) If equations are on their own line, use the math directive ' \
                  '"```{math}\n:label: equation-label \n .... \n```".' \
                  ' Always use labels in directives and ' \
                  'reference them in the text using [](label) format. Only reference using the link, ' \
                  'do not include e.g. Eq. beforehand. Use existing labels if they are present, otherwise ' \
                  'make the label at least 10 characters long and ' \
                  'unique to avoid duplicates - do not use eq1, eq2 etc. ' \
                  '(3) Exercises and solutions. Use the exercise ' \
                  'directive "````{exercise} \n:label: exercise-label \n ..... \n````" for exercises ' \
                  'and worked examples, ONLY using the label option. ' \
                  'Use the solution directive ' \
                  '"````{solution} exercise-label \n:class: dropdown \n ..... \n````" for ' \
                  'solutions. ' \
                  '(4) Images/figures in the format ' \
                  '"```{figure} filename\n:width: %s \n:name: figure-label\n:alt: %s\nCaption\n```". ' \
                  'Never use .pdf images. Always replace .pdf or .eps filenames with .png. %s' \
                  '(5) Tables in the format ' \
                  '":::{table} Caption\n:widths: auto\n:align: center\n:name: table-label\n markdown ' \
                  'table \n:::".' \
                  ' ' % (width_comment, alt_text_comment, image_comment)

    if mode == 'summary':

        llm_system_prompt = 'Your role is to extract information from lecture notes. ' \
                            'Extract the lecturer name in the format <author:>. If there are multiple lecturers, ' \
                            'includes all of them. ' \
                            'Extract the course name in the format <course:>. ' \
                            'Extract the course code in the format <code:>. ' \
                            'If there is an abstract, extract it format <abstract:>. ' \
                            'Return only this information and nothing else. ' \
                            'If you cannot find the information do not include it.'

    elif mode == 'tex_to_md':

        llm_system_prompt = 'Your role is to convert latex strings into accessible HTML. To do this convert ' \
                            'LaTeX layout to Markdown. If a chapter or section is labelled, first link to it by ' \
                            '"(heading-label)=\n", using EXACTLY the same label as the latex document, ' \
                            'then on the next line use chapters at heading level 1 (#), ' \
                            'sections at heading level 2 (##), subsections at heading level 3 (###), ' \
                            'and subsubsections at heading level 4 (####). You can then link to sections ' \
                            'by [](heading-label). ' \
                            'If a chapter or section is not labelled, do not link to it, and put ' \
                            'chapters at heading level 1 (#), ' \
                            'sections at heading level 2 (##), subsections at heading level 3 (###), ' \
                            'and subsubsections at heading level 4 (####).' \
                            'For citations use the format "{footcite}`citation-name`". ' \
                            'Always enclose inline mathematical expressions in $...$ format. '
        llm_system_prompt += myst_prompt
        llm_system_prompt += 'Use normal markdown for everything else. '
        llm_system_prompt += 'You will be provided a string in latex format and you should ONLY ' \
                             'output the direct conversion. Convert the ENTIRE document.' \
                             ' NEVER output \\ref{...}, instead using the ' \
                             'correct referencing. '

    elif mode == 'pdf_to_md':

        llm_system_prompt = 'Your role is to convert the document into accessible HTML.  To do this convert ' \
                            'the ENTIRE document to Markdown. For sections (##) do NOT include the X.Y text, ' \
                            'for subsections (###) do NOT include the X.Y.Z text. ' \
                            'Always enclose inline mathematical expressions in $...$ format. '
        llm_system_prompt += myst_prompt
        llm_system_prompt += 'Use normal markdown for everything else. '
        llm_system_prompt += 'You will be provided a document and you should ONLY ' \
                             'output the direct conversion. Convert the ENTIRE document in the order it is on ' \
                             'the page. Do NOT put the conversion in a ```{markdown} ... ``` block. '

    elif mode == 'pdf_to_tex':

        if figure_paths is None:
            figure_comment = ''
        else:
            figure_comment = 'The figure filenames to use in the order figures appear on the page ' \
                             'are ' + ', '.join(figure_paths) + ' .'

        llm_system_prompt = 'Convert the document to latex. You will be provided a document and you should ONLY ' \
                            'output the direct conversion. Convert the ENTIRE document in the order it is on ' \
                            'the page. You can use the amsmath package. ' \
                            'Do NOT put the conversion in a ```latex ... ``` block. ' \
                            'Determine the chapter by referring to headings and equations. ' \
                            'Headings with a SINGLE number ONLY (e.g. 1..., 2.... ) are \\chapters. ' \
                            'Headings with 2 numbers of the form X.Y (e.g. 1.1) are \\sections.' \
                            ' Headings with 3 numbers of the form X.Y.Z ' \
                            '(e.g. 1.1.1) are \\subsections. ' \
                            '%s' % figure_comment

    else:
        raise ValueError

    return llm_system_prompt


def main(args):
    api_key = os.environ.get('OPENAI_API_KEY', args.openai_api_key)

    if not os.path.isdir(args.out):
        urllib.request.urlretrieve("https://github.com/adammoss/ExampleBook/archive/main.zip", filename="main.zip")
        shutil.unpack_archive("main.zip")
        os.remove("main.zip")
        shutil.move("ExampleBook-main", args.out)
        os.system("jupyter-book build %s" % args.out)
        shutil.copy("%s/Figures/UoNTransparent.png" % args.out, "%s/_build/html/_static/." % args.out)
        shutil.copy("%s/Figures/UoNTransparentDark.png" % args.out, "%s/_build/html/_static/." % args.out)
        os.remove("%s/Chapters/test.md" % args.out)
        shutil.rmtree("%s/Chapters/plots" % args.out)

    with open(os.path.join(args.out, "_config.yml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(args.out, "_toc.yml"), "r") as f:
        toc = yaml.load(f, Loader=yaml.FullLoader)

    client = OpenAI(api_key=api_key)

    functions = []
    if args.accessibility:
        functions.append(signature_get_accessibility)

    def complete(messages, functions=None):

        if functions is None:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
            )
        else:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                functions=functions,
                function_call="auto",
            )
        total_tokens = resp.usage.total_tokens
        print('Tokens used: %s ' % total_tokens)
        message = resp.choices[0].message

        if hasattr(message, "function_call") and message.function_call is not None:
            print(message.function_call.arguments)
            fn_args = json.loads(message.function_call.arguments)
            fn_response, fn_tokens_used = execute_function_call(message.function_call.name, fn_args, args, api_key)
            print(fn_response)
            total_tokens += fn_tokens_used
            messages.append({"role": "function", "content": fn_response, "name": message.function_call.name})
        else:
            messages.append({"role": "assistant", "content": message.content})

        return messages, total_tokens

    total_tokens_used = 0
    start_time = time.time()

    file_path = args.in_file

    os.makedirs(os.path.join(args.out, "tmp"), exist_ok=True)

    md_files = []

    if '.pdf' in file_path and os.path.isfile(file_path):

        # PDF conversion to tex

        print('Converting %s' % file_path)

        doc = fitz.open(file_path)

        page = doc[0]

        image_path = os.path.join(args.out, "tmp", 'doc-%s.png' % 0)
        page.get_pixmap(dpi=200).save(image_path)

        output, total_tokens = transcribe_image(image_path, api_key, get_system_prompt(mode="summary"))
        total_tokens_used += total_tokens

        course_name = re.search('.*?<course:(.*?)>.*', output)
        if course_name is not None:
            title = course_name.group(1).strip()
        else:
            title = ""
        author = ', '.join(re.findall('<author:(.*?)>', output))

        print('-' * 50)

        page_output = []

        for i, page in enumerate(doc, start=1):
            if i < args.min_page or i > args.max_page - 1:
                continue
            image_path = os.path.join(args.out, "tmp", 'doc-%s.png' % i)
            page_path = os.path.join(args.out, "tmp", 'doc-%s.tex' % i)
            figure_paths = []
            for idx, img in enumerate(page.get_images(), start=1):
                data = doc.extract_image(img[0])
                with PIL.Image.open(io.BytesIO(data.get('image'))) as im:
                    figure_paths.append(f'{i}-{idx}.{data.get("ext")}')
                    im.save(os.path.join(args.out, "tmp", f'{i}-{idx}.{data.get("ext")}'), mode='wb')

            if not args.force and os.path.exists(page_path):
                print('Markdown file exists for page %s. Set -f to force conversion.' % i)
                with open(page_path) as f:
                    output = f.read()
            else:
                print('Converting page %s/%s' % (i, len(doc)))
                page.get_pixmap(dpi=200).save(image_path)
                output, total_tokens = transcribe_image(image_path, api_key,
                                                        get_system_prompt(mode="pdf_to_tex",
                                                                          accessibility=args.accessibility,
                                                                          figure_paths=figure_paths),
                                                        vision_model=args.vision_model,
                                                        )
                total_tokens_used += total_tokens
                if '\\begin{document}' in output:
                    output = output.split('\\begin{document}')[1]
                    if '\\end{document}' in output:
                        output = output.split('\\end{document}')[0]

            with open(page_path, 'w') as f:
                f.write(output)

            page_output.append(output)
            file_path = os.path.join(args.out, "tmp", "doc.tex")
            full_output = ['\\documentclass{report}\n\\usepackage{amsmath}\n\\author{%s}\n\\title{%s}\n\\begin'
                           '{document}' % (author, title)] + page_output + ['\n\\end{document}']
            with open(file_path, 'w') as f:
                f.write('\n\n'.join(full_output))

    if '.tex' in file_path and os.path.isfile(file_path):

        # Tex conversion to markdown

        path = Path(file_path)
        parent_path = path.parent.absolute()

        print('Converting %s' % file_path)

        with open(file_path) as f:
            content = f.read()

        # Insert any input files
        input_paths = re.findall(r'\\input\{(.*)\}', content)
        for input_path in input_paths:
            if os.path.isfile(os.path.join(str(parent_path), input_path + '.tex')):
                with open(os.path.join(str(parent_path), input_path + '.tex')) as f:
                    input_content = f.read()
            elif os.path.isfile(os.path.join(str(parent_path), input_path)):
                with open(os.path.join(str(parent_path), input_path)) as f:
                    input_content = f.read()
            else:
                continue
            content = content.replace('\\input{' + input_path + '}', input_content)

        # Find any pdf images in tex files
        image_paths = re.findall(r'\\includegraphics\[.*\]\{(.*?)\}', content)
        image_paths += re.findall(r'\\includegraphics\{(.*?)\}', content)
        for image_path in image_paths:
            os.makedirs(Path(os.path.join(args.out, "Chapters", image_path)).parent.absolute(), exist_ok=True)
            if '.pdf' in image_path and os.path.exists(os.path.join(str(parent_path), image_path)):
                # This converts PDF images to PNG
                doc = fitz.open(os.path.join(str(parent_path), image_path))
                doc[0].get_pixmap(dpi=200).save(os.path.join(args.out, "Chapters",
                                                             image_path.replace('.pdf', '.png')))
            elif '.eps' in image_path and os.path.exists(os.path.join(str(parent_path), image_path)):
                # This converts .eps images to PNG
                if os.path.exists(
                        os.path.join(str(parent_path), image_path.replace('.eps', '-eps') + '-converted-to.pdf')):
                    doc = fitz.open(os.path.join(str(parent_path), image_path.replace('.eps', '-eps') +
                                                 '-converted-to.pdf'))
                    doc[0].get_pixmap(dpi=200).save(os.path.join(args.out, "Chapters",
                                                                 image_path.replace('.eps', '.png')))
                else:
                    try:
                        image = PIL.Image.open(os.path.join(str(parent_path), image_path))
                        image.load(scale=10)
                        image.save(os.path.join(args.out, "Chapters", image_path.replace('.eps', '.png')))
                    except:
                        pass
            elif os.path.exists(os.path.join(str(parent_path), image_path)):
                # Otherwise just copy
                shutil.copy(os.path.join(str(parent_path), image_path),
                            os.path.join(args.out, "Chapters", image_path))

        # Find any bib files
        bibtex_bibfiles = []
        for bib_path in re.findall(r'\\bibliography\{(.*?)\}', content):
            if os.path.exists(os.path.join(str(parent_path), bib_path)):
                shutil.copy(os.path.join(str(parent_path), bib_path), args.out)
                bibtex_bibfiles.append(bib_path)
            if os.path.exists(os.path.join(str(parent_path), bib_path + '.bib')):
                shutil.copy(os.path.join(str(parent_path), bib_path + '.bib'), args.out)
                bibtex_bibfiles.append(bib_path + '.bib')
        if len(bibtex_bibfiles) > 0:
            config["bibtex_bibfiles"] = bibtex_bibfiles

        for replacement in re.findall(r'\\def(.*?)\{(.*)\}', content):
            if 'equation' in replacement[1] or 'eqnarray' in replacement[1]:
                continue
            # Regex doesn't quite work
            #content = re.sub(r'%s([$\\\n, =_^\.])' % re.escape(replacement[0]), r'%s' % re.escape(replacement[1]) + r'\1',
            #                 content)
            for char in [',', '\\', '\n', ' ', '=', '_', '^', '.', '$', '+', '-']:
                content = content.replace(replacement[0] + char, replacement[1] + char)

        new_defs = re.findall(r'(\\newcommand.*?)\n', content)

        content_list = re.split(r'(\\chapter\*?{.*}|[^}]\\section\*?{.*}|\\subsection\*?{.*}|\\subsubsection\*?{.*})',
                                content)

        if 'feynmp' in content:
            functions.append(signature_latex)

        # Assume the course metadata is contained within first content block

        messages = [
            {"role": "system", "content": get_system_prompt(mode="summary")},
            {"role": "user", "content": "Extract information from the following: %s" % content_list[0]}
        ]

        messages, total_tokens = complete(messages)
        total_tokens_used += total_tokens

        output = messages[-1]["content"]

        course_name = re.search('.*?<course:(.*?)>.*', output)
        if course_name is not None:
            config["title"] = course_name.group(1).strip()
        else:
            config["title"] = ""
        config["author"] = ', '.join(re.findall('<author:(.*?)>', output))
        course_code = re.search('.*?<code:(.*?)>.*', output)
        if course_code is not None:
            course_code = course_code.group(1).strip()
        else:
            course_code = ""
        course_abstract = re.search('.*?<abstract:(.*?)>.*', output)
        if course_abstract is not None:
            course_abstract = course_abstract.group(1).strip()
        else:
            course_abstract = ""

        print('-' * 50)
        print('Title: %s' % config["title"])
        print('Code: %s' % course_code)
        print('Author(s): %s' % config["author"])
        print('-' * 50)

        with open(os.path.join(args.out, "_config.yml"), "w") as f:
            yaml.dump(config, f)

        main_md = "# %s: %s \n\n" \
                  "Welcome to %s. %s \n\n " \
                  "```{note}\nThese notes are in HTML format for improved accessibility and support for " \
                  "assistive technologies. If you have any comments on them, please contact %s (module convener) or " \
                  "[email Dr Adam Moss](mailto:adam.moss@nottingham.ac.uk) (Digital Learning Lead)" \
                  "\n```\n\n```{tableofcontents}\n```" % \
                  (course_code, config["title"], config["title"], course_abstract, config["author"])
        out_file = os.path.join(args.out, "main.md")
        with open(out_file, 'w') as f:
            f.write(main_md)

        # Combine small chunks
        token_count = 0
        chunk = []
        chunks = []
        split_section = '\\section'
        for i, content in enumerate(content_list[1:]):
            if content == '':
                continue
            if '\\chapter' in content:
                split_section = '\\chapter'
            token_count += num_tokens(content, model=args.model)
            chunk.append(content)
            if token_count > args.concat_token_count or i == len(content_list) - 1:
                chunks.append('\n'.join(chunk))
                token_count = 0
                chunk = []

        chapter_output = []
        chapter_count = 0

        for i, chunk in enumerate(chunks):

            output = None
            if os.path.exists(os.path.join(args.out, "tmp", "chunk_%s.tex" % i)):
                with open(os.path.join(args.out, "tmp", "chunk_%s.tex" % i)) as f:
                    chunk_compare = f.read()
                if not args.force and chunk == chunk_compare and \
                        os.path.exists(os.path.join(args.out, "tmp", "chunk_%s.md" % i)):
                    print('Markdown file exists for identical chunk %s. Set -f to force conversion.' % i)
                    with open(os.path.join(args.out, "tmp", "chunk_%s.md" % i)) as f:
                        output = f.read()
                else:
                    with open(os.path.join(args.out, "tmp", "chunk_%s.tex" % i), 'w') as f:
                        f.write(chunk)
            else:
                with open(os.path.join(args.out, "tmp", "chunk_%s.tex" % i), 'w') as f:
                    f.write(chunk)

            if output is None or (isinstance(args.force_chunks, list) and i in args.force_chunks):

                print('Converting chunk %s/%s with token length %s: %s' % (i, len(chunks) - 1,
                                                                           num_tokens(chunk, args.model),
                                                                           chunk[0:50].replace('\n', '')))

                messages = [
                    {"role": "system", "content": get_system_prompt(mode="tex_to_md",
                                                                    accessibility=args.accessibility)},
                ]
                if len(new_defs) > 0:
                    messages.append({
                        "role": "system",
                        "content": "The following definitions convert "
                                   "non-standard latex commands: %s. "
                                   "ALWAYS make these replacements whenever you see one.  " % '\n'.join(new_defs)
                    })
                messages.append({
                    "role": "system", "content": "If the latex contains feynmf commands, obtain the outermost block "
                                                 "of nested environments that contains feynmf commands. "
                                                 "Ensure you include the entire set of nested environments, "
                                                 "it may be inside an equation array. "
                                                 "Use this in the available function call to generate the diagram."
                })
                if args.extra_prompt:
                    messages.append({"role": "system", "content": args.extra_prompt})
                if '\\includegraphics' in chunk:
                    image_count = chunk.count('\\includegraphics')
                    messages.append({
                        "role": "user",
                        "content": "Convert the following. There are %s images present so use the "
                                   "function call for each: %s" % (image_count, chunk)
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": "Convert the following: %s" % chunk
                    })

                if len(functions) > 0:
                    messages, total_tokens = complete(messages, functions=functions)
                else:
                    messages, total_tokens = complete(messages)
                total_tokens_used += total_tokens

                call_count = 0
                while messages[-1]["role"] == "function":
                    call_count = call_count + 1
                    if call_count < 5:
                        messages, total_tokens = complete(messages, functions=functions)
                    else:
                        messages, total_tokens = complete(messages)
                    total_tokens_used += total_tokens

                output = messages[-1]["content"]

                with open(os.path.join(args.out, "tmp", "chunk_%s.md" % i), 'w') as f:
                    f.write(output)

            if split_section in chunk and len(chapter_output) > 0:
                chapter_count += 1
                out_file = os.path.join(args.out, "Chapters", "chapter_%s.md" % chapter_count)
                if len(bibtex_bibfiles) > 0:
                    # Add foot bib https://github.com/executablebooks/jupyter-book/issues/1997#issuecomment-1590145393
                    chapter_output.append("```{footbibliography}\n```")
                with open(out_file, 'w') as f:
                    f.write('\n\n'.join(chapter_output))
                md_files.append(os.path.join("Chapters", "chapter_%s.md" % chapter_count))
                chapter_output = []
                toc["chapters"] = [{"file": md_file} for md_file in md_files]
                with open(os.path.join(args.out, "_toc.yml"), "w") as f:
                    yaml.dump(toc, f)
                os.system("jupyter-book build %s" % args.out)

            chapter_output.append(output)

            print('-' * 50)

        if len(chapter_output) > 0:
            chapter_count += 1
            out_file = os.path.join(args.out, "Chapters", "chapter_%s.md" % chapter_count)
            if len(bibtex_bibfiles) > 0:
                # Add foot bib https://github.com/executablebooks/jupyter-book/issues/1997#issuecomment-1590145393
                chapter_output.append("```{footbibliography}\n```")
            with open(out_file, 'w') as f:
                f.write('\n\n'.join(chapter_output))
            md_files.append(os.path.join("Chapters", "chapter_%s.md" % chapter_count))
            toc["chapters"] = [{"file": md_file} for md_file in md_files]
            with open(os.path.join(args.out, "_toc.yml"), "w") as f:
                yaml.dump(toc, f)
            os.system("jupyter-book build --all %s" % args.out)

    print("Total tokens used: %s" % total_tokens_used)
    print("Time taken (seconds): %s" % round((time.time() - start_time)))


def run_script(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str)
    parser.add_argument('-m', '--model', type=str, default='gpt-4-1106-preview')
    parser.add_argument('-v', '--vision_model', type=str, default='gpt-4-vision-preview')
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-k', '--openai_api_key', type=str, default=None)
    parser.add_argument('-o', '--out', type=str, default='Book')
    parser.add_argument('-a', '--accessibility', default=False, action='store_true')
    parser.add_argument('-f', '--force', default=False, action='store_true')
    parser.add_argument('-p', '--extra_prompt', type=str, default=None)
    parser.add_argument('--concat_token_count', type=int, default=100)
    parser.add_argument('--min_page', type=int, default=0)
    parser.add_argument('--max_page', type=int, default=1000)
    parser.add_argument('--force_chunks', type=int, action='append')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_script()
