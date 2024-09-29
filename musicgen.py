import os
import time
import torch
import typing
import shutil
import subprocess
import gradio as gr
import pandas as pd
from pathlib import Path
from einops import rearrange
from tempfile import NamedTemporaryFile
from concurrent.futures import ProcessPoolExecutor

from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.models.encodec import InterleaveStereoCompressionModel

import warnings
warnings.filterwarnings("ignore")

MBD = None
MODEL = None
USE_MBD = False
INTERRUPTING = False
WAV_RATE = 32000
WAV_CHANNELS = 1
MUSIC_CSV = "output/output.csv"

# 减少日志打印
_old_call = subprocess.call
def _new_call(*args, **kwargs):
    kwargs['stderr'] = subprocess.DEVNULL
    kwargs['stdout'] = subprocess.DEVNULL
    _old_call(*args, **kwargs)
subprocess.call = _new_call

# 加载模型
def load_model(model_name):
    global MODEL
    if MODEL is None or MODEL.name != model_name:
        del MODEL
        torch.cuda.empty_cache()
        print("Loading Model", model_name)
        MODEL = MusicGen.get_pretrained(model_name)

# 加载解码器
def load_MBD():
    global MBD
    if MBD is None:
        print("loading MBD")
        MBD = MultiBandDiffusion.get_mbd_musicgen()

# 中断生成
def interrupt():
    global INTERRUPTING
    INTERRUPTING = True

# 音频转视频动画
def make_waveform(*args, **kwargs):
    start_time = time.time()
    kwargs['animate'] = True
    out = gr.make_waveform(*args, **kwargs)
    tt = time.time() - start_time
    print("音频转动画耗时", tt)
    return out

# 控制解码器输出显示
def visible_MBD(decoder_choice):
    if decoder_choice == "MultiBandDiffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2

# 文件清理器
class FileCleaner: # 定时清理文件
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []
    def add(self, path: typing.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))
    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break
file_cleaner = FileCleaner(file_lifetime=3600)

def predict(text, melody, duration, progress=False, gradio_progress=None, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    if melody is not None:
        sr, melody_wav = melody
        melody_wav = torch.from_numpy(melody_wav).to(MODEL.device).float().t()
        if melody_wav.dim() == 1:
            melody_wav = melody_wav.unsqueeze(0)
        melody_wav = melody_wav[..., :int(sr * duration)] # 裁剪音频
        melody_wav = convert_audio(melody_wav, sr, WAV_RATE, WAV_CHANNELS)
        # 文本 + 旋律 ==> 音乐
        outputs = MODEL.generate_with_chroma(
            [text], progress=progress, return_tokens=USE_MBD,
            melody_wavs=[melody_wav], melody_sample_rate=WAV_RATE)
    else:
        # 文本 ==> 音乐
        outputs = MODEL.generate(
            [text], progress=progress, return_tokens=USE_MBD)
    # 使用MBD
    if USE_MBD:
        if gradio_progress is not None:
            gradio_progress(1, desc='Running MBD...')
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
            left, right = MODEL.compression_model.get_left_right_codes(outputs[1])
            outputs_MBD = MBD.tokens_to_wav(torch.cat([left, right]))
            outputs_MBD = rearrange(outputs_MBD, '(s b) c t -> b (s c) t', s=2)
        else:
            outputs_MBD = MBD.tokens_to_wav(outputs[1])
        outputs = torch.cat([outputs[0], outputs_MBD], dim=0)
    # 音乐写入音频文件和视频文件
    with ProcessPoolExecutor(4) as pool:
        out_wavs = []
        video_pool = []
        for output in outputs.detach().cpu().float():
            with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
                audio_write(
                    file.name, output, MODEL.sample_rate, strategy="loudness",
                    loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
                video_pool.append(pool.submit(make_waveform, file.name))
                file_cleaner.add(file.name)
                out_wavs.append(file.name)
        out_videos = [video.result() for video in video_pool]
    for video in out_videos: file_cleaner.add(video);
    print("缓存文件数量", len(file_cleaner.files))
    return out_wavs, out_videos

def generate(model_path, model_choice, decoder_choice, text, audio, duration, cfg_coef, temper, top_p, top_k, progress=gr.Progress()):
    if not text.strip(): return;
    global USE_MBD, INTERRUPTING
    INTERRUPTING = False
    USE_MBD = False
    # 加载模型
    progress(0, desc="Loading model...")
    if model_path:
        if not Path(model_path).exists():
            raise gr.Error(f"Model path {model_path} does not exist.")
        if not Path(model_path).is_dir():
            raise gr.Error(f"Model path {model_path} must be folder.")
        load_model(model_path)
    else:
        load_model("models/musicgen/" + model_choice)
    # 加载解码器
    if decoder_choice == "MultiBandDiffusion":
        progress(0, desc="Loading MBD...")
        USE_MBD = True
        load_MBD()
    # 生成进度条
    max_generated = 0
    def progress_callback(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        generated = min(max_generated, to_generate)
        progress((generated, to_generate), desc="Running model...")
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(progress_callback)
    # 翻译文本为英文
    from translators import translate_text
    text2 = translate_text(text)
    print("【text】", text, "-->", text2)
    # 生成音乐并返回
    try:
        wavs, videos = predict(
            text2, audio, duration, progress=True, gradio_progress=progress,
            temperature=temper, top_p=top_p, top_k=int(top_k), cfg_coef=cfg_coef)
    except gr.Error as e:
        print("【Error】", e)
        return None, None, None, None, None
    if not USE_MBD:
        return text[:10], wavs[0], videos[0], None, None
    else:
        return text[:10], wavs[0], videos[0], wavs[1], videos[1]

def save_video(df: pd.DataFrame, name: str, video, video_MBD):
    save_path = "output/" + name + ".mp4"
    if video_MBD:
        shutil.copy(video_MBD, save_path)
        print(save_path, "已成功保存！")
    elif video:
        shutil.copy(video, save_path)
        print(save_path, "已成功保存！")
    else:
        print("视频不存在，无法保存！")
        return df 
    df.loc[len(df)] = [len(df)+1, name+".mp4"]
    df.to_csv(MUSIC_CSV, index=False)
    return df

def del_video(df: pd.DataFrame, num: int):
    if num==0: return df; # 返回原表格
    file_name = df.loc[num-1, "音乐"]
    os.remove(f"output/{file_name}")
    print(file_name, "已成功删除！")
    df = df.drop(num-1).reset_index(drop=True)
    df["序号"] = df.index + 1 # 序号=索引+1
    df.to_csv(MUSIC_CSV, index=False)
    return df


with gr.Blocks(fill_width=True) as demo:
    gr.Markdown("<h1 style='text-align: center; font-size: 2em'>音乐生成</h1>")
    with gr.Row():
        # 参数列
        with gr.Column(min_width=0, scale=1):
            model_path = gr.Textbox(
                label="模型路径", placeholder="选填", lines=1, max_lines=1, interactive=True, show_copy_button=True)
            model_choice = gr.Radio(
                label="Model Choice", value="musicgen-small",
                choices=[c for c in os.listdir("models/musicgen")], interactive=True)
            decoder_choice = gr.Radio(
                label="Decoder Choice", value="Default",
                choices=["Default", "MultiBandDiffusion"], interactive=True)
            duration = gr.Slider(label="音乐时长", minimum=1, maximum=100, step=1, value=10, interactive=True)
            cfg_coef = gr.Slider(label="风格强度", minimum=0, maximum=10, step=0.1, value=5, interactive=True)
            temper = gr.Slider(label="Temper", minimum=0, maximum=1, step=0.1, value=1.0, interactive=True)
            top_p = gr.Slider(label="top_P", minimum=0, maximum=100, step=1.0, value=0.0, interactive=True)
            top_k = gr.Slider(label="top_K", minimum=0, maximum=500, step=10, value=250, interactive=True)
        # 输入列
        with gr.Column(min_width=0, scale=2):
            input_text = gr.Textbox(
                label="Input Text", placeholder="必填", lines=1, max_lines=1, interactive=True, show_copy_button=True, autofocus=True)
            input_audio = gr.Audio(label="Input Audio", interactive=True, show_download_button=True)
            with gr.Row():
                interrupt_btn = gr.Button("中断").click(fn=interrupt)
                generate_btn = gr.Button("生成", variant="primary")
            gr.Examples(
                inputs=[input_text, input_audio, model_choice, decoder_choice],
                examples=[
                    ["吉他, 民谣", None, "musicgen-small", "Default"],
                    ["鼓声, 悲壮", None, "musicgen-small", "Default"],
                    ["钢琴, 古典", None, "musicgen-small", "MultiBandDiffusion"],
                    ["笛声, 悠扬", "examples/纯1.mp3", "musicgen-songstarter-v0.1", "Default"],
                    ["古筝, 欢快", "examples/纯2.mp3", "musicgen-songstarter-v0.2", "Default"],
                ],
            )
        # 输出列
        with gr.Column(min_width=0, scale=1):
            music_name = gr.Textbox(
                label="音乐名称", placeholder="十个字以内", lines=1, max_lines=1, interactive=True, show_copy_button=True, max_length=10)
            output_audio = gr.Audio(label="Output Audio", interactive=False)
            output_video = gr.Video(label="Output Vidio", interactive=False, height=150)
            output_audio_MBD = gr.Audio(label="Output Audio (MBD)", interactive=False, visible=False)
            output_video_MBD = gr.Video(label="Output Vidio (MBD)", interactive=False, visible=False, height=150)
            save_btn = gr.Button("保存")
    
    # 音乐列表
    with gr.Accordion(label="🎵 音乐列表", open=False):
        with gr.Row():
            # 列表列
            with gr.Column(min_width=0, scale=1):
                DF = gr.Dataframe(value=pd.read_csv(MUSIC_CSV), show_label=False, interactive=False, column_widths=[1,3])
                with gr.Row():
                    @gr.render(inputs=DF)
                    def del_col(df: pd.DataFrame):
                        del_num = gr.Number(min_width=0, scale=3, show_label=False, minimum=0, maximum=len(df))
                        del_btn = gr.Button(min_width=0, scale=1, value="删除", variant="stop")
                        del_btn.click(fn=del_video, inputs=[DF, del_num], outputs=DF)
            # 播放列
            with gr.Column(min_width=0, scale=3):
                with gr.Row():
                    @gr.render(inputs=DF)
                    def create_videos(DF: pd.DataFrame):
                        for file in DF["音乐"]:
                            file_path = f"output/{file}"
                            if os.path.exists(file_path):
                                file_name = file.split('.')[0]
                                gr.Video(label=file_name, value=file_path, height=100, min_width=210, interactive=False, loop=True)
    # 响应事件
    decoder_choice.change(visible_MBD, decoder_choice, [output_audio_MBD, output_video_MBD], show_progress=False)
    generate_btn.click(
        fn=generate,
        inputs=[model_path, model_choice, decoder_choice, input_text, input_audio, duration, cfg_coef, temper, top_p, top_k],
        outputs=[music_name, output_audio, output_video, output_audio_MBD, output_video_MBD])
    input_text.submit(
        fn=generate,
        inputs=[model_path, model_choice, decoder_choice, input_text, input_audio, duration, cfg_coef, temper, top_p, top_k],
        outputs=[music_name, output_audio, output_video, output_audio_MBD, output_video_MBD])
    save_btn.click(
        fn=save_video,
        inputs=[DF, music_name, output_video, output_video_MBD],
        outputs=DF)

demo.launch(inbrowser=True, share=False)