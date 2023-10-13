import subprocess
from platform import system
from os import listdir
import tempfile
import typer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = typer.Typer()

MAX_LENGTH = 512
INPUT_LENGTH = MAX_LENGTH * 3

isLinux = system() == "Linux"

def formatPath(path):
  if (isLinux):
    return path.replace(" ", "\\ ")
  else:
    return f"\"{path}\""
  
def add_trailing_slash(path: str):
  if (path.endswith("/")):
    return ""
  return "/"

def translate_func(
  srt_f: str,
  output_file: str,
  src_lang: str,
  target_lang: str,
):
  tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}")
  model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}")

  tokens = []
  result = ""
  with open(srt_f, "r") as f:
    lines = f.readlines()
    newInput = ""
    lines_in_input = 0
    nlines = len(lines)
    for line_index in range(1, nlines):
      lines_in_input += 1
      line = lines[line_index - 1]
      fline = line.replace("\n", "[NL]")
      if (lines_in_input <= 12):
        newInput += fline
      if (lines_in_input > 12 or line_index == (nlines - 1)):
        lines_in_input = 0
        tokens.append(
          tokenizer(
            newInput,
            return_tensors="pt",
            additional_special_tokens=["[NL]"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
          ).input_ids
        )
        newInput = fline
  for token in tokens:
    outputs = model.generate(input_ids=token, num_beams=5, num_return_sequences=3)
    fragment = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    result = result + fragment[0]
  result = result.replace("\n\n\n", "\n\n")
  with open(output_file, "w") as f:
    f.write(result.replace("[NL]","\n"))

def transcribe(
  file: str,
  output_dir: str,
  src_lang: str,
  model: str,
  translate: bool = False,
  target_lang: str = "es",
):
  file = formatPath(file)
  output_dir = formatPath(output_dir)
  output = ""
  if (translate):
    output = output_dir
    output_dir = tempfile.mkdtemp()
  subprocess.run(
    f"whisper {file} --model {model} --language {src_lang} --output_dir {output_dir} --output_format srt",
    shell=True
  )
  if (translate):
    file_name = listdir(output_dir)[0]
    output = f"{output}{add_trailing_slash(output)}{file_name}"
    translate_func(
      srt_f=f"{output_dir}{add_trailing_slash(output_dir)}{file_name}",
      output_file=output,
      src_lang=src_lang,
      target_lang=target_lang,
    )
    print("temp transcribe dir", output_dir)
    print("temp translate dir", output)
  

def add_subs(
  video: str,
  srt_f: str,
  output_file: str,
):
  video = formatPath(video)
  srt_f = formatPath(srt_f)
  output_file = formatPath(output_file)
  subprocess.run(
    f"ffmpeg -i {video} -vf subtitles={srt_f} {output_file}",
    shell=True
  )

@app.command("transcribe", options_metavar="translate")
def sample_func(
  file: str,
  output_dir: str,
  src_lang: str,
  model: str = "tiny",
  translate: bool = False,
  target_lang: str = "es",
):
  print("start")
  print(file)
  print(output_dir)
  transcribe(
    file=file,
    output_dir=output_dir,
    src_lang=src_lang,
    model=model,
    translate=translate,
    target_lang=target_lang,
  )
  print("end")

@app.command("translate")
def sample_func(
  srt_f: str,
  output_file: str,
  src_lang: str,
  target_lang: str,
):
  translate_func(
    srt_f=srt_f,
    output_file=output_file,
    src_lang=src_lang,
    target_lang=target_lang,
  )


@app.command("add-subs")
def add_subs_command(
  video: str,
  srt_f: str,
  output_file: str,
):
  add_subs(
    video=video,
    srt_f=srt_f,
    output_file=output_file,
  )
  
@app.command("transcribe-to-video")
def transcribe_subs(
  file: str,
  output_file: str,
  src_lang: str,
  model: str = "tiny",
  translate: bool = False,
  target_lang: str = "es",
):
  srt_dir = tempfile.mkdtemp()
  transcribe(
    file=file,
    output_dir=srt_dir,
    src_lang=src_lang,
    model=model,
    translate=translate,
    target_lang=target_lang
  )
  print(srt_dir)
  print(listdir(srt_dir))
  file_name = listdir(srt_dir)[0]
  srt_f = f"{srt_dir}{add_trailing_slash(srt_dir)}{file_name}"
  add_subs(
    video=file,
    srt_f=srt_f,
    output_file=output_file
  )

if __name__ == "__main__":
  app()
