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
    print(image_path)
    try:
        base64_image = encode_image(image_path)
    except FileNotFoundError:
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
    message = resp['choices'][0]['message']
    recommended_width = re.search('.*?<(\d+)px>.*', message['content'])
    alt_text = re.sub('<(\d+)px>', '', message['content']).replace(':', '-')
    result = 'Alt text: %s' % alt_text
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


def execute_function_call(function_name, args):
    fn_tokens_used = 0
    if function_name == "get_accessibility":
        content, fn_tokens_used = get_accessibility(**args)
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
    if mode == 'summary':

        llm_system_prompt = 'Your role is to extract information from lecture notes. ' \
                            'Extract the lecturer name in the format <author:>. If there are multiple lecturers, ' \
                            'includes all of them. ' \
                            'Extract the course name in the format <course:>. ' \
                            'Extract the course code in the format <code:>. ' \
                            'Return only this information and nothing else. ' \
                            'If you cannot find the information do not include it.'

    elif mode == 'tex':

        if accessibility:
            image_comment = 'ALWAYS use the function call to get image accessibility information (alt text and width)' \
                            ', whenever there is a figure or graphics. Do NOT use the caption as alt text.'
            alt_text_comment = 'ONLY insert the returned alt text from the available function call.'
            width_comment = 'Use the recommended width from the available function call, otherwise set to 500px.'
        else:
            image_comment = ''
            alt_text_comment = 'alt text'
            width_comment = '500px'

        llm_system_prompt = 'Your role is to convert latex strings into accessible HTML. To do this convert ' \
                            'LaTeX layout to Markdown, using chapters at heading level 1 (#), ' \
                            'sections at heading level 2 (##), subsections at heading level 3 (###), ' \
                            'and subsubsections at heading level 4 (####). ' \
                            'Always enclose inline mathematical expressions in $...$ format. ' \
                            'Use MyST ONLY in the following cases: ' \
                            '(1) Definitions in the format "````{admonition} .... \n:class: .... \n .... \n````". ' \
                            '(2) If equations are on their own line, use the math directive ' \
                            '"```{math}\n:label: \n .... \n```".' \
                            ' Always use labels in directives and ' \
                            'reference them in the text using [](label) format. Only reference using the link, ' \
                            'do not include e.g. Eq. beforehand. Make the label at least 10 characters long and ' \
                            'unique to avoid duplicates - do not use eq1, eq2 etc. ' \
                            '(3) Exercises and solutions. Use the exercise ' \
                            '"directive ````{exercise} \n:label: exercise-label \n ..... \n````" for exercises ' \
                            'and worked examples, ONLY using the label option. ' \
                            'Use the solution directive ' \
                            '"directive ````{solution} exercise-label \n:class: dropdown \n ..... \n````" for ' \
                            'solutions. ' \
                            '(4) Images/figures in the format ' \
                            '"```{figure} filename\n:width: %s \n:name: figure-label\n:alt: %s\nCaption\n```". ' \
                            'Never use .pdf images. Always replace .pdf filenames with .png. %s' \
                            '(5) Tables in the format ' \
                            '":::{table} Caption\n:widths: auto\n:align: center\n:name: table-label\n markdown ' \
                            'table \n:::".' \
                            ' ' % (width_comment, alt_text_comment, image_comment)
        llm_system_prompt += 'Use normal markdown for everything else. '
        llm_system_prompt += 'You will be provided a string in latex format and you should ONLY ' \
                             'output the direct conversion. '

    else:

        if figure_paths is None:
            figure_comment = ''
        else:
            figure_comment = 'The figure filenames to use in the order figures appear on the page ' \
                             'are ' + ', '.join(figure_paths) + ' .'

        llm_system_prompt = 'Your role is to convert the document into accessible HTML.  To do this convert ' \
                            'the ENTIRE document to Markdown. For sections (##) do NOT include the X.Y text, ' \
                            'for subsections (###) do NOT include the X.Y.Z text. ' \
                            'Always enclose inline mathematical expressions in $...$ format. ' \
                            'Use MyST ONLY in the following cases: ' \
                            '(1) Definitions in the format "````{admonition} .... \n:class: .... \n .... \n````". ' \
                            '(2) If equations are on their own line, use the math directive ' \
                            '"```{math}\n:label: \n .... \n```". ' \
                            ' Always use labels in directives and ' \
                            'reference them in the text using [](label) format. Only reference using the link, ' \
                            'do not include e.g. Eq. beforehand. Make the label at least 10 characters long and ' \
                            'unique to avoid duplicates - do not use eq1, eq2 etc.  ' \
                            '(3) Exercises and solutions. Use the exercise ' \
                            '"directive ````{exercise} \n:label: exercise-label \n ..... \n````" for exercises ' \
                            'and worked examples, ONLY using the label option. ' \
                            'Use the solution directive ' \
                            '"directive ````{solution} exercise-label \n:class: dropdown \n ..... \n````" for ' \
                            'solutions. ' \
                            '(4) Images/figures in the format ' \
                            '"```{figure} filename\n:width: ..px \n:name: figure-label\n:alt: ..\nCaption\n```". %s' \
                            '(5) Tables in the format ' \
                            '":::{table} Caption\n:widths: auto\n:align: center\n:name: table-label\n markdown ' \
                            'table \n:::". ' % figure_comment
        llm_system_prompt += 'Use normal markdown for everything else. '
        llm_system_prompt += 'You will be provided a document and you should ONLY ' \
                             'output the direct conversion. Convert the ENTIRE document in the order it is on ' \
                             'the page. Do NOT put the conversion in a ```{markdown} ... ``` block. '

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

    def complete(messages, function_call: str = "auto"):

        if function_call is None:
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
                functions=[signature_get_accessibility],
                function_call="auto",
            )
        total_tokens = resp.usage.total_tokens
        print('Tokens used: %s ' % total_tokens)
        message = resp.choices[0].message

        if hasattr(message, "function_call") and message.function_call is not None:
            fn_args = json.loads(message.function_call.arguments)
            fn_args["api_key"] = api_key
            fn_args["image_path"] = os.path.join(args.out, "Chapters", fn_args["image_path"])
            fn_args["vision_model"] = args.vision_model
            if '.pdf' in fn_args["image_path"]:
                fn_args["image_path"] = fn_args["image_path"].replace('.pdf', '.png')
            fn_response, fn_tokens_used = execute_function_call(message.function_call.name, fn_args)
            print(fn_response)
            total_tokens += fn_tokens_used
            messages.append({"role": "function", "content": fn_response, "name": message.function_call.name})
        else:
            messages.append({"role": "assistant", "content": message.content})

        return messages, total_tokens

    total_tokens_used = 0
    start_time = time.time()

    chapter_output = []
    md_files = []
    chapter_count = 0

    file_path = args.in_file

    os.makedirs(os.path.join(args.out, "tmp"), exist_ok=True)

    if '.tex' in file_path and os.path.isfile(file_path):

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
        image_paths = re.findall(r'\\includegraphics\[.*\]\{(.*)\}', content)
        image_paths += re.findall(r'\\includegraphics\{(.*)\}', content)
        for image_path in image_paths:
            if 'pdf' in image_path and os.path.exists(os.path.join(str(parent_path), image_path)):
                # This converts PDF images to PNG
                doc = fitz.open(os.path.join(str(parent_path), image_path))
                os.makedirs(Path(os.path.join(args.out, "Chapters", image_path)).parent.absolute(), exist_ok=True)
                doc[0].get_pixmap(dpi=200).save(os.path.join(args.out, "Chapters",
                                                             image_path.replace('.pdf', '.png')))
            elif os.path.exists(os.path.join(str(parent_path), image_path)):
                # Otherwise just copy
                os.makedirs(Path(os.path.join(args.out, "Chapters", image_path)).parent.absolute(), exist_ok=True)
                shutil.copy(os.path.join(str(parent_path), image_path),
                            os.path.join(args.out, "Chapters", image_path))
        content_list = re.split(r'(\\chapter\*?{.*}|\\section\*?{.*}|\\subsection\*?{.*}|\\subsubsection\*?{.*})',
                                content)

        # Assume the course metadata is contained within first content block

        messages = [
            {"role": "system", "content": get_system_prompt(mode="summary")},
            {"role": "user", "content": "Extract information from the following: %s" % content_list[0]}
        ]

        messages, total_tokens = complete(messages, function_call="none")
        total_tokens_used += total_tokens

        output = messages[-1]["content"]

        course_name = re.search('.*?<course:(.*?)>.*', output)
        if course_name is not None:
            config["title"] = course_name.group(1).strip()
            print('Title: %s' % config["title"])
        else:
            config["title"] = ""
        author = re.search('.*?<author:(.*?)>.*', output)
        if author is not None:
            config["author"] = author.group(1).strip()
            print('Author: %s' % config["author"])
        else:
            config["author"] = ""

        main_md = "# %s \n\n" \
                  "Welcome to %s. \n\n " \
                  "```{note}\nThese notes are in HTML format for improved accessibility and support for " \
                  "assistive technologies. If you have any comments on them, please contact %s\n```\n\n " \
                  "```{tableofcontents}\n```" % (config["title"], config["title"], config["author"])
        out_file = os.path.join(args.out, "main.md")
        with open(out_file, 'w') as f:
            f.write(main_md)

        print('-' * 50)

        # Combine small chunks
        token_count = 0
        chunk = []
        chunks = []
        for i, content in enumerate(content_list[1:]):
            if content == '':
                continue
            token_count += num_tokens(content, model=args.model)
            chunk.append(content)
            if token_count > args.concat_token_count or i == len(content_list) - 1:
                chunks.append('\n'.join(chunk))
                token_count = 0
                chunk = []

        for i, chunk in enumerate(chunks):

            if not args.force and os.path.exists(os.path.join(args.out, "tmp", "chunk_%s.md" % i)):
                print('Markdown file exists for chunk %s. Set -f to force conversion.' % i)
                with open(os.path.join(args.out, "tmp", "chunk_%s.md" % i)) as f:
                    output = f.read()

            else:

                print('Converting chunk %s with token length %s: %s' % (i + 1, num_tokens(chunk, args.model),
                                                                        chunk[0:50].replace('\n', '')))

                messages = [
                    {"role": "system", "content": get_system_prompt(mode="tex", accessibility=args.accessibility)},
                ]
                if 'includegraphics' in chunk:
                    messages.append({
                        "role": "user",
                        "content": "Convert the following. There are images present so use the "
                                   "function call: %s" % chunk
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": "Convert the following: %s" % chunk
                    })

                if args.accessibility:
                    messages, total_tokens = complete(messages)
                else:
                    messages, total_tokens = complete(messages, function_call="none")
                total_tokens_used += total_tokens

                call_count = 0
                while messages[-1]["role"] == "function":
                    call_count = call_count + 1
                    if call_count < 5:
                        messages, total_tokens = complete(messages)
                    else:
                        messages, total_tokens = complete(messages, function_call="none")
                    total_tokens_used += total_tokens

                output = messages[-1]["content"]

                with open(os.path.join(args.out, "tmp", "chunk_%s.md" % i), 'w') as f:
                    f.write(output)

            if ('\\chapter' in chunk or i == len(chunks) - 1) and len(chapter_output) > 0:
                if i == len(chunks) - 1:
                    chapter_output.append(output)
                chapter_count += 1
                out_file = os.path.join(args.out, "Chapters", "chapter_%s.md" % chapter_count)
                with open(out_file, 'w') as f:
                    f.write('\n\n'.join(chapter_output))
                md_files.append(os.path.join("Chapters", "chapter_%s.md" % chapter_count))
                chapter_output = []

            chapter_output.append(output)

            print('-' * 50)

    elif '.pdf' in file_path and os.path.isfile(file_path):

        path = Path(file_path)
        parent_path = path.parent.absolute()

        print('Converting %s' % file_path)

        doc = fitz.open(file_path)

        page = doc[0]

        image_path = os.path.join(args.out, "tmp", 'doc-%s.png' % 0)
        page.get_pixmap(dpi=200).save(image_path)

        output, total_tokens = transcribe_image(image_path, api_key, get_system_prompt(mode="summary"))
        total_tokens_used += total_tokens

        course_name = re.search('.*?<course:(.*?)>.*', output)
        if course_name is not None:
            config["title"] = course_name.group(1).strip()
            print('Title: %s' % config["title"])
        else:
            config["title"] = ""
        author = re.search('.*?<author:(.*?)>.*', output)
        if author is not None:
            config["author"] = author.group(1).strip()
            print('Author: %s' % config["author"])
        else:
            config["author"] = ""

        main_md = "# %s \n\n" \
                  "Welcome to %s. \n\n " \
                  "```{note}\nThese notes are in HTML format for improved accessibility and support for " \
                  "assistive technologies. If you have any comments on them, please contact %s\n```\n\n " \
                  "```{tableofcontents}\n```" % (config["title"], config["title"], config["author"])
        out_file = os.path.join(args.out, "main.md")
        with open(out_file, 'w') as f:
            f.write(main_md)

        print('-' * 50)

        for i, page in enumerate(doc, start=1):
            if i < args.min_page or i > args.max_page - 1:
                continue
            image_path = os.path.join(args.out, "tmp", 'doc-%s.png' % i)
            page_path = os.path.join(args.out, "tmp", 'doc-%s.md' % i)
            figure_paths = []
            for idx, img in enumerate(page.get_images(), start=1):
                data = doc.extract_image(img[0])
                with PIL.Image.open(io.BytesIO(data.get('image'))) as im:
                    figure_paths.append(f'{i}-{idx}.{data.get("ext")}')
                    im.save(os.path.join(args.out, "Chapters", f'{i}-{idx}.{data.get("ext")}'), mode='wb')

            if not args.force and os.path.exists(page_path):
                print('Markdown file exists for page %s. Set -f to force conversion.' % i)
                with open(page_path) as f:
                    output = f.read()
            else:
                print('Converting %s' % image_path)
                page.get_pixmap(dpi=200).save(image_path)
                output, total_tokens = transcribe_image(image_path, api_key,
                                                        get_system_prompt(mode="pdf",
                                                                          accessibility=args.accessibility,
                                                                          figure_paths=figure_paths),
                                                        vision_model=args.vision_model,
                                                        )

            with open(page_path, 'w') as f:
                f.write(output)
            chapter_output.append(output)
            total_tokens_used += total_tokens
            out_file = os.path.join(args.out, "Chapters", "chapter_%s.md" % 1)
            with open(out_file, 'w') as f:
                f.write('\n\n'.join(chapter_output))
            md_files = [os.path.join("Chapters", "chapter_%s.md" % 1)]

    toc["chapters"] = [{"file": md_file} for md_file in md_files]
    with open(os.path.join(args.out, "_toc.yml"), "w") as f:
        yaml.dump(toc, f)
    with open(os.path.join(args.out, "_config.yml"), "w") as f:
        yaml.dump(config, f)
    os.system("jupyter-book build %s" % args.out)

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
    parser.add_argument('--concat_token_count', type=int, default=100)
    parser.add_argument('--min_page', type=int, default=0)
    parser.add_argument('--max_page', type=int, default=1000)
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_script()