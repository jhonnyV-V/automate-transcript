# Automate-Transcribe

a small python project to automatically create transcriptions (and translate them) of videos an audio 

## Requirements
1. Python 3.10
2. Ffmpeg
```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```
3. Install the requirements in requirements.txt
```bash
pip install -r requirement.txt
```
4. If you have a gpu and want to use it, install the [correct](https://pytorch.org/get-started/locally/) pytorch version
```bash
#example
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## How to use
this projects use whisper, so everytime a model parameters appear this are the accepted values
| Multilingual model |
|:------------------:|
|       `tiny`       |
|       `base`       |
|      `small`       |
|      `medium`      |
|      `large`       |
|      `large-v2`    |
|      `large-v3`    |

### Transcribe command
the transcribe command should output a .srt file in the output_dir


the transcribe command have some optional parameters
|      Name     |   Type    |  Default Value  |
|:-------------:|:---------:|:---------------:| 
|  gpu          |  boolean  |  false          |    
|  translate    |  boolean  |  false          |   
|  target_lang  |  string   |  es             |

and this are the positional parameters in order
|      Name    |   Type    |
|:------------:|:---------:|
|  file        |  string   |
|  output_dir  |  string   |
|  src_lang    |  string   |
|  model       |  string   |

```bash
python main.py transcribe "C:\Users\username\Videos\Music Videos\Ren - Hi Ren.mp4" "C:\Users\username\Videos\Music Videos" en --model "large-v3" --gpu --translate
```

### Translate command
the translate command should output a .srt file in the output_dir

and this are the positional parameters in order
|      Name    |   Type    |
|:------------:|:---------:|
|  srt_path    |  string   |
|  output_dir  |  string   |
|  src_lang    |  string   |
|  model       |  string   |

```bash
python main.py translate "~/Videos/Music Videos/Ren - Depression.srt" "~/Videos/Music Videos/Ren - Depression es.srt" en es
```

### Add Subs command
the add subs command takes .srt and mp4 and outputs a mp4 video

and this are the positional parameters in order
|      Name    |   Type    |
|:------------:|:---------:|
|  video       |  string   |
|  srt_path    |  string   |
|  output_file |  string   |

```bash
python main.py add-subs "D:/input/Ren - Hi Ren.mp4" "D:/output/Ren - Hi Ren es.srt" "D:/output/Ren - Hi Ren.mp4"
```

### Transcribe To Video command
the transcribe to video command create the transcript of a video and add the subtitles to the video, it also can translate the transcript.

and this are the positional parameters in order
|      Name    |   Type    |
|:------------:|:---------:|
|  video       |  string   |
|  output_file |  string   |
|  src_lang    |  string   |

optional parameters
|      Name     |   Type    |  Default Value  |
|:-------------:|:---------:|:---------------:| 
|  gpu          |  boolean  |  false          |    
|  translate    |  boolean  |  false          |   
|  target_lang  |  string   |  es             |


```bash
python main.py transcribe-to-video "D:/money game/Ren - Animal Flow.mp4" "D:/money game/output/" en --model medium
```

## Fully soported languages
|  Languages      |   Code    |
|:---------------:|:---------:|
|  English        |  en       |
|  Chinese        |  zh       |
|  German         |  de       |
|  Spanish        |  es       |
|  Russian        |  ru       |
|  Korean         |  ko       |
|  French         |  fr       |
|  Japanese       |  ja       |
|  Portuguese     |  pt       |
|  Turkish        |  tr       |
|  Polish         |  pl       |
|  Catalan        |  ca       |
|  Dutch          |  nl       |
|  Arabic         |  ar       |
|  Swedish        |  sv       |
|  Italian        |  it       |
|  Indonesian     |  id       |
|  Hindi          |  hi       |
|  Finnish        |  fi       |
|  Vietnamese     |  vi       |
|  Hebrew         |  he       |
|  Ukrainian      |  uk       |
|  Greek          |  el       |
|  Malay          |  ms       |
|  Czech          |  cs       |
|  Romanian       |  ro       |
|  Danish         |  da       |
|  Hungarian      |  hu       |
|  Tamil          |  ta       |
|  Norwegian      |  no       |
|  Thai           |  th       |
|  Urdu           |  ur       |
|  Croatian       |  hr       |
|  Bulgarian      |  bg       |
|  Lithuanian     |  lt       |
|  Welsh          |  cy       |
|  Slovak         |  sk       |
|  Telugu         |  te       |
|  Persian        |  fa       |
|  Latvian        |  lv       |
|  Bengali        |  bn       |
|  Serbian        |  sr       |
|  Azerbaijani    |  az       |
|  Slovenian      |  sl       |
|  Kannada        |  kn       |
|  Estonian       |  et       |
|  Macedonian     |  mk       |
|  Basque         |  eu       |
|  Icelandic      |  is       |
|  Armenian       |  hy       |
|  Nepali         |  ne       |
|  Mongolian      |  mn       |
|  Bosnian        |  bs       |
|  Kazakh         |  kk       |
|  Swahili        |  sw       |
|  Galician       |  gl       |
|  Marathi        |  mr       |
|  Punjabi        |  pa       |
|  Khmer          |  km       |
|  Shona          |  sn       |
|  Yoruba         |  yo       |
|  Somali         |  so       |
|  Afrikaans      |  af       |
|  Occitan        |  oc       |
|  Georgian       |  ka       |
|  Belarusian     |  be       |
|  Tajik          |  tg       |
|  Sindhi         |  sd       |
|  Lao            |  lo       |
|  Uzbek          |  uz       |
|  Pashto         |  ps       |
|  Maltese        |  mt       |
|  Luxembourgish  |  lb       |
|  Tagalog        |  tl       |
|  Assamese       |  as       |
|  Javanese       |  jw       |
|  Cantonese      |  yue      |
