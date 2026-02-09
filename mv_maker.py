# ========== 核心修复：指定Gradio本地临时目录，解决PermissionError权限问题 ==========
import os
import tempfile

# 临时文件存项目内的gradio_temp文件夹，避开系统权限目录，自动创建
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.path.dirname(__file__), "gradio_temp")
os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)
# ==================================================================================

# ===================== 前置补丁+依赖导入 =====================
import PIL.Image as Image

if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# 核心依赖（仅保留基础必用）
import librosa
import numpy as np
import gradio as gr
import shutil
import uuid
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
from PIL import ImageDraw, ImageFont

# 确保项目内临时目录存在（自动创建）
os.makedirs("temp_output", exist_ok=True)
os.makedirs("temp_text", exist_ok=True)
os.makedirs("temp_subtitles", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# 全局变量（仅保留视频路径，极简）
generated_video_path = None


# ===================== 核心工具函数（完全保留，功能不变） =====================
def parse_pos(pos_str, base_size, elem_size=0, is_x=True):
    """解析位置：精准居中计算（核心保留）"""
    pos_str = pos_str.strip().lower()
    if pos_str.isdigit():
        return int(pos_str)
    key = ""
    offset = 0
    for i, c in enumerate(pos_str):
        if c.isdigit():
            key = pos_str[:i]
            offset = int(pos_str[i:])
            break
    if not key:
        key = pos_str

    if key == "center":
        return (base_size / 2) - (elem_size / 2)
    if is_x:
        if key == "left":
            return offset
        elif key == "right":
            return base_size - offset - elem_size
        else:
            return offset
    else:
        if key == "top":
            return offset
        elif key == "bottom":
            return base_size - offset - elem_size
        else:
            return offset


def parse_color(color_str):
    """解析颜色：#FFFFFF/red/rgb等（核心保留）"""
    color_str = color_str.strip().lower()
    preset_colors = {
        "white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0),
        "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
        "orange": (255, 165, 0), "purple": (128, 0, 128), "gray": (128, 128, 128)
    }
    if color_str in preset_colors:
        return preset_colors[color_str]
    try:
        hex_color = color_str.lstrip('#')
        if len(hex_color) == 6:
            return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
    except:
        pass
    try:
        if color_str.startswith("rgb(") and color_str.endswith(")"):
            rgb_part = color_str[4:-1].split(",")
            return (int(rgb_part[0].strip()), int(rgb_part[1].strip()), int(rgb_part[2].strip()))
    except:
        pass
    return (255, 255, 255)


def detect_voice_segments(audio_path, threshold=0.02, min_duration=0.3):
    """语音段检测（核心保留，你的12段语音正常检测）"""
    if not os.path.exists(audio_path):
        return "❌ 音频文件不存在，请重新上传！", []
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = 2048
    hop_length = 512
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(energy, sr=sr, hop_length=hop_length)
    voice_frames = energy > threshold
    segments = []
    start = None
    for i, is_voice in enumerate(voice_frames):
        if is_voice and start is None:
            start = times[i]
        elif not is_voice and start is not None:
            end = times[i]
            if end - start >= min_duration:
                segments.append((round(start, 2), round(end, 2)))
            start = None
    if start is not None:
        end = times[-1]
        if end - start >= min_duration:
            segments.append((round(start, 2), round(end, 2)))
    # 合并相邻短语音段
    final_segments = []
    for seg in segments:
        if not final_segments:
            final_segments.append(seg)
        else:
            last_s, last_e = final_segments[-1]
            if seg[0] - last_e < 0.2:
                final_segments[-1] = (last_s, seg[1])
            else:
                final_segments.append(seg)
    if not final_segments:
        return "❌ 未检测到语音，请调低阈值重试！", []
    # 格式化检测结果
    tip = f"✅ 检测到{len(final_segments)}个语音段：\n"
    for i, (s, e) in enumerate(final_segments, 1):
        tip += f"{i}. {s}秒 → {e}秒（时长：{e - s:.2f}秒）\n"
    tip += "\n💡 请按语音段数输入对应行数的纯字幕！"
    return tip, final_segments


def match_subtitle_with_voice(subtitle_text, voice_segments, start_offset=0.0, end_offset=0.0):
    """字幕匹配语音段（核心保留，时间轴防重叠）"""
    if not voice_segments:
        return "❌ 请先检测语音段！"
    subtitle_lines = [line.strip() for line in subtitle_text.strip().split("\n") if line.strip()]
    if not subtitle_lines:
        return "❌ 请输入纯字幕文本（每行一段）！"

    matched_lines = []
    last_end_time = 0.0
    for i, line in enumerate(subtitle_lines):
        if i < len(voice_segments):
            s, e = voice_segments[i]
            s = round(s + start_offset, 2)
            e = round(e + end_offset, 2)
            s = max(0.0, s)
            s = max(s, last_end_time)  # 强制防重叠
            e = max(s + 0.5, e)  # 最小时长0.5秒
        else:
            # 字幕行数超语音段，自动生成连续时间
            avg_dur = np.mean([e - s for s, e in voice_segments]) if voice_segments else 3.0
            s = round(last_end_time, 2)
            e = round(s + avg_dur, 2)
        last_end_time = e
        # 生成标准格式字幕
        matched_lines.append(f"{s},{line},{e},36,#FFFFFF,center,bottom100")
    return "\n".join(matched_lines)


def parse_subtitles(subtitle_text, video_w, video_h):
    """解析字幕配置（核心保留）"""
    subtitles = []
    if not subtitle_text.strip():
        return subtitles
    lines = subtitle_text.strip().split("\n")
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            parts = line.split(",")
            pos_y_str = parts[-1].strip() if len(parts) >= 1 else "bottom100"
            pos_x_str = parts[-2].strip() if len(parts) >= 2 else "center"
            color_str = parts[-3].strip() if len(parts) >= 3 else "#FFFFFF"
            font_size_str = parts[-4].strip() if len(parts) >= 4 else "36"
            end_time_str = parts[-5].strip() if len(parts) >= 5 else "0"
            content_parts = parts[1:-5] if len(parts) >= 6 else [f"字幕{idx + 1}"]
            start_time_str = parts[0].strip() if len(parts) >= 6 else "0"

            start_time = float(start_time_str) if start_time_str else 0.0
            end_time = float(end_time_str) if end_time_str else 0.0
            start_time = max(0.0, round(start_time, 2))
            end_time = max(start_time + 0.5, round(end_time, 2))

            font_size = int(font_size_str) if font_size_str.isdigit() else 36
            font_size = max(10, min(100, font_size))

            content = ",".join([p.strip() for p in content_parts]).strip() or f"字幕{idx + 1}"
            color = parse_color(color_str)

            subtitles.append({
                "start": start_time, "end": end_time, "content": content,
                "font_size": font_size, "color": color,
                "pos_x_str": pos_x_str, "pos_y_str": pos_y_str
            })
        except Exception as e:
            raise gr.Error(
                f"第{idx + 1}行字幕解析失败：{str(e)}\n✅ 正确格式：0.0,你好世界,3.0,36,#FFFFFF,center,bottom100")
    return subtitles


def create_text_image(text, size, color, bg_color=(0, 0, 0, 0)):
    """生成字幕/文字图片（核心保留，精准居中）"""
    try:
        # 兼容Windows/Mac/Linux字体
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # Windows黑体
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "/Library/Fonts/Arial.ttf",  # Mac
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, size)
                    break
                except:
                    continue
        if font is None:
            font = ImageFont.load_default()
        # 计算文字宽高（兼容新旧PIL）
        dummy_img = Image.new('RGBA', (1, 1), bg_color)
        draw = ImageDraw.Draw(dummy_img)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            w, h = draw.textsize(text, font=font)
        # 生成透明背景文字图片
        img = Image.new('RGBA', (w, h), bg_color)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font, fill=color)
        temp_path = os.path.join("temp_text", f"text_{uuid.uuid4()}.png")
        img.save(temp_path, format='PNG')
        return temp_path, w, h
    except Exception as e:
        raise Exception(f"文字生成失败：{str(e)}")


def create_slideshow_clip(img_paths, duration, slide_duration=3.0):
    """多张背景图轮播核心函数（完全保留，你的核心需求）"""
    if len(img_paths) == 0:
        raise Exception("❌ 请至少上传一张背景图！")
    if len(img_paths) == 1:
        # 单张图直接显示全程
        return ImageClip(img_paths[0]).set_duration(duration)

    # 多张图自动均分时长，最后一张补全剩余时间，避免时长不匹配
    num_imgs = len(img_paths)
    base_duration = duration / num_imgs
    clip_list = []
    remaining_duration = duration

    for i, img_path in enumerate(img_paths):
        if i == num_imgs - 1:
            img_dur = remaining_duration
        else:
            img_dur = min(base_duration, slide_duration)
            remaining_duration -= img_dur

        img_clip = ImageClip(img_path).set_duration(img_dur)
        clip_list.append(img_clip)

    # 拼接轮播剪辑，兼容不同尺寸图片
    slideshow_clip = concatenate_videoclips(clip_list, method="compose")
    return slideshow_clip


def mp3_images_to_mp4(mp3_path, img_paths, slide_duration, text="", text_size=30, text_color="#FFFFFF",
                      text_pos="center,80", watermark_path=None, watermark_alpha=0.5,
                      watermark_pos="right20,bottom20", subtitle_text=""):
    """核心合成：MP3+多张背景轮播+字幕+水印（你的核心需求）"""
    global generated_video_path
    temp_files = []
    try:
        # 基础校验
        if not os.path.exists(mp3_path):
            raise Exception(f"❌ MP3文件不存在：{mp3_path}")
        if not img_paths or len(img_paths) == 0:
            raise Exception("❌ 请至少上传一张背景图！")

        # 加载音频，获取总时长
        audio = AudioFileClip(mp3_path)
        audio_duration = audio.duration

        # 1. 创建背景轮播剪辑（核心功能，多张图自动切换）
        bg_clip = create_slideshow_clip(img_paths, audio_duration, slide_duration)
        vid_w, vid_h = bg_clip.size

        # 2. 生成全局文字（全程显示）
        main_text_clip = None
        if text.strip():
            rgb = parse_color(text_color)
            text_img_path, text_w, text_h = create_text_image(text, text_size, rgb)
            temp_files.append(text_img_path)
            main_text_clip = ImageClip(text_img_path).set_duration(audio_duration)
            tx_str, ty_str = text_pos.split(",") if "," in text_pos else (text_pos, "0")
            tx = parse_pos(tx_str, vid_w, text_w, is_x=True)
            ty = parse_pos(ty_str, vid_h, text_h, is_x=False)
            main_text_clip = main_text_clip.set_position((tx, ty))

        # 3. 生成水印（可选）
        watermark_clip = None
        if watermark_path and os.path.exists(watermark_path):
            wm_img = Image.open(watermark_path)
            w, h = wm_img.size
            # 等比例缩放水印到高度80px
            new_w = int(w * (80 / h))
            new_h = 80
            wm_img_resized = wm_img.resize((new_w, new_h),
                                           Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BILINEAR)
            wm_temp = os.path.join("temp_subtitles", f"wm_{uuid.uuid4()}.png")
            temp_files.append(wm_temp)
            wm_img_resized.save(wm_temp)
            # 生成水印剪辑
            watermark_clip = ImageClip(wm_temp).set_opacity(watermark_alpha).set_duration(audio_duration)
            wx_str, wy_str = watermark_pos.split(",") if "," in watermark_pos else (watermark_pos, "0")
            wx = parse_pos(wx_str, vid_w, new_w, is_x=True)
            wy = parse_pos(wy_str, vid_h, new_h, is_x=False)
            watermark_clip = watermark_clip.set_position((wx, wy))

        # 4. 生成精准字幕（时间轴防重叠，居中显示）
        subtitle_clips = []
        if subtitle_text.strip():
            subtitles = parse_subtitles(subtitle_text, vid_w, vid_h)
            for sub in subtitles:
                if sub["end"] > audio_duration:
                    sub["end"] = audio_duration
                # 生成字幕图片
                sub_img_path, sub_w, sub_h = create_text_image(sub["content"], sub["font_size"], sub["color"])
                temp_files.append(sub_img_path)
                # 生成字幕剪辑
                sub_clip = ImageClip(sub_img_path).set_duration(sub["end"] - sub["start"])
                sub_x = parse_pos(sub["pos_x_str"], vid_w, sub_w, is_x=True)
                sub_y = parse_pos(sub["pos_y_str"], vid_h, sub_h, is_x=False)
                sub_clip = sub_clip.set_position((sub_x, sub_y)).set_start(sub["start"])
                subtitle_clips.append(sub_clip)

        # 组合所有视频轨：背景轮播→全局文字→水印→字幕
        all_clips = [bg_clip]
        if main_text_clip:
            all_clips.append(main_text_clip)
        if watermark_clip:
            all_clips.append(watermark_clip)
        all_clips.extend(subtitle_clips)
        final_clip = CompositeVideoClip(all_clips).set_audio(audio)

        # 导出MP4视频（H264编码，兼容性强）
        output_path = os.path.join("temp_output", f"mv_{uuid.uuid4()}.mp4")
        final_clip.write_videofile(
            output_path, codec="libx264", audio_codec="aac",
            fps=15, threads=4, verbose=False, logger=None
        )

        # 清理临时文件，释放资源
        audio.close()
        final_clip.close()
        for f in temp_files:
            try:
                os.remove(f)
            except:
                pass
        generated_video_path = output_path
        return output_path
    except Exception as e:
        # 异常时也清理临时文件
        for f in temp_files:
            try:
                os.remove(f)
            except:
                pass
        raise gr.Error(f"MV生成失败：{str(e)}")


def download_video():
    """下载生成的MV"""
    global generated_video_path
    if generated_video_path and os.path.exists(generated_video_path):
        return generated_video_path
    else:
        raise gr.Error("❌ 请先生成MV后再下载！")


# ===================== Gradio界面（3.0最早期版本兼容，无任何高版本组件） =====================
# 极简CSS美化（适配低版本，仅保留基础好看的样式）
custom_css = """
/* 主按钮渐变美化 */
.gradio-container .button-primary {
    background: linear-gradient(135deg, #2a93b7 0%, #d94691 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
/* 按钮hover效果 */
.gradio-container .button-primary:hover {
    opacity: 0.9 !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.1) !important;
}
/* 所有输入框圆角 */
.gradio-container input, .gradio-container textarea, .gradio-container .slider {
    border-radius: 6px !important;
    border: 1px solid #e2e8f0 !important;
}
/* 标题文字美化 */
.gradio-container h1 {
    color: #2a93b7 !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin-bottom: 20px !important;
}
.gradio-container h2, .gradio-container h3 {
    color: #334155 !important;
    font-weight: 600 !important;
    margin-top: 15px !important;
}
/* 整体容器内边距 */
.gradio-container {
    padding: 20px !important;
}
"""

# 构建基础界面（仅用Tabs/Row/Column/基础组件，无任何高级组件）
with gr.Blocks(title="🎤 AI翻唱MV生成器（轮播版）", css=custom_css) as demo:
    # 顶部主标题
    gr.Markdown("# 🎤 AI翻唱MV生成器（多张背景轮播版）")
    gr.Markdown("### ✨ 核心功能：多张背景图轮播 | 字幕精准居中 | 语音段自动检测 | 时间轴防重叠")
    gr.Markdown("---")  # 用markdown横线替代Divider，兼容低版本

    # 隐藏状态变量：存储语音段（替代全局变量，防止数据叠加）
    voice_segments_state = gr.State(value=[])

    # 标签页（仅用基础TabItem，无icon）
    with gr.Tabs():
        # 标签1：音频上传与语音检测
        with gr.TabItem("音频语音检测"):
            gr.Markdown("## 🎵 上传MP3音频并检测语音段")
            mp3_input = gr.Audio(label="上传翻唱MP3音频", type="filepath")
            detect_threshold = gr.Slider(
                label="语音检测阈值（越小越灵敏，杂音多调至0.03-0.04）",
                minimum=0.01, maximum=0.1, value=0.02, step=0.01
            )
            detect_btn = gr.Button("🔍 开始检测语音段", variant="primary")
            voice_result = gr.Textbox(
                label="语音检测结果", lines=6,
                placeholder="检测结果将显示在这里，会列出所有语音段的起止时间..."
            )
            gr.Markdown("---")

        # 标签2：字幕输入与时间轴匹配
        with gr.TabItem("字幕时间轴生成"):
            gr.Markdown("## ✏️ 输入纯字幕并匹配语音时间轴")
            pure_subtitle = gr.Textbox(
                label="纯字幕文本（每行一段，行数与语音段数一致，无时间）",
                lines=8,
                placeholder="示例：\n生活就像一杯清茶\n初入口时或许有些苦涩\n但细细品味\n却能感受到其中的甘甜与清香"
            )
            gr.Markdown("### ⏱️ 字幕时间全局偏移")
            with gr.Row():
                global_start_offset = gr.Slider(
                    label="开始偏移（±秒）：负数=提前显示，正数=延后显示",
                    minimum=-1.0, maximum=1.0, value=0.0, step=0.1
                )
                global_end_offset = gr.Slider(
                    label="结束偏移（±秒）：负数=提前隐藏，正数=延后隐藏",
                    minimum=-1.0, maximum=1.0, value=0.0, step=0.1
                )
            match_btn = gr.Button("⚡ 一键匹配语音时间轴", variant="primary")
            matched_subtitle = gr.Textbox(
                label="匹配后的带时间轴字幕（可手动微调）", lines=10,
                placeholder="匹配后将生成标准格式：开始时间,内容,结束时间,字号,颜色,水平位置,垂直位置..."
            )
            gr.Markdown("---")

        # 标签3：核心功能 - 多张背景轮播 + MV生成/下载
        with gr.TabItem("MV生成与导出"):
            with gr.Row():
                # 左侧：配置区（轮播/文字/水印）
                with gr.Column(scale=2):
                    # 背景轮播核心配置（你的核心需求）
                    gr.Markdown("## 🖼️ 多张背景图轮播设置")
                    bg_imgs = gr.File(
                        label="上传多张背景图（支持批量选择，JPG/PNG均可）",
                        file_count="multiple", file_types=["image"]
                    )
                    slide_duration = gr.Slider(
                        label="单张图片显示时长（秒），图片多建议设1-2秒",
                        minimum=1.0, maximum=10.0, value=3.0, step=0.5
                    )
                    gr.Markdown("---")

                    # 全局文字配置
                    gr.Markdown("## 📜 全局文字（视频全程显示）")
                    global_text = gr.Textbox(label="文字内容", placeholder="AI翻唱MV | 轮播版 | 字幕精准居中")
                    global_text_size = gr.Slider(label="文字大小", minimum=10, maximum=100, value=30, step=1)
                    global_text_color = gr.ColorPicker(label="文字颜色", value="#FFFFFF")
                    global_text_pos = gr.Textbox(
                        label="文字位置（示例：center,80 水平居中+距上80px | right20,bottom50 右下20px）",
                        value="center,80"
                    )
                    gr.Markdown("---")

                    # 水印配置（可选）
                    gr.Markdown("## 🔖 水印设置（可选，建议PNG透明背景）")
                    watermark_img = gr.Image(label="上传水印图片", type="filepath")
                    wm_alpha = gr.Slider(label="水印透明度", minimum=0.1, maximum=1.0, value=0.5, step=0.1)
                    wm_pos = gr.Textbox(label="水印位置（示例：right20,bottom20）", value="right20,bottom20")
                    gr.Markdown("---")

                    # 操作按钮
                    with gr.Row():
                        generate_btn = gr.Button("🚀 生成MV", variant="primary")
                        download_btn = gr.Button("📥 下载MV")

                # 右侧：字幕微调 + 预览下载
                with gr.Column(scale=3):
                    gr.Markdown("## 🎯 最终字幕配置（可手动修改时间/样式）")
                    final_subtitle = gr.Textbox(
                        label="最终字幕（匹配后自动同步，可手动改）",
                        lines=12, value=""
                    )
                    gr.Markdown("---")
                    gr.Markdown("## 🎥 MV预览与下载")
                    video_output = gr.Video(label="生成的MV（轮播背景+精准字幕）", height=400)
                    download_output = gr.File(label="下载的MP4视频文件")

    # ===================== 绑定所有交互事件（基础绑定，兼容低版本） =====================
    # 检测语音段
    detect_btn.click(
        fn=detect_voice_segments,
        inputs=[mp3_input, detect_threshold],
        outputs=[voice_result, voice_segments_state]
    )
    # 匹配字幕时间轴
    match_btn.click(
        fn=match_subtitle_with_voice,
        inputs=[pure_subtitle, voice_segments_state, global_start_offset, global_end_offset],
        outputs=matched_subtitle
    )
    # 匹配后的字幕自动同步到最终字幕框
    matched_subtitle.change(
        fn=lambda x: x,
        inputs=matched_subtitle,
        outputs=final_subtitle
    )
    # 生成MV（核心：传入多张背景图+轮播时长）
    generate_btn.click(
        fn=mp3_images_to_mp4,
        inputs=[mp3_input, bg_imgs, slide_duration, global_text, global_text_size, global_text_color,
                global_text_pos, watermark_img, wm_alpha, wm_pos, final_subtitle],
        outputs=video_output
    )
    # 下载MV
    download_btn.click(
        fn=download_video,
        inputs=[],
        outputs=download_output
    )

# ===================== 启动应用 + 自动清理 + 依赖安装 =====================
if __name__ == "__main__":
    # 自动安装缺失依赖（清华源加速，解决下载慢/失败）
    required_pkgs = ["gradio", "moviepy", "pillow", "librosa", "numpy"]
    for pkg in required_pkgs:
        try:
            __import__(pkg)
        except ImportError:
            print(f"正在安装缺失依赖：{pkg}")
            os.system(f"pip install {pkg} -i https://pypi.tuna.tsinghua.edu.cn/simple")

    # 启动Gradio服务（本地访问，端口7860）
    print("✅ 服务启动成功，浏览器访问：http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # 关闭公网分享，仅本地使用
    )


    # 程序退出时自动清理所有临时文件，避免占用磁盘
    def cleanup_temp_files():
        for dir_name in ["temp_output", "temp_text", "temp_subtitles", "temp_audio", "gradio_temp"]:
            if os.path.exists(dir_name):
                try:
                    shutil.rmtree(dir_name)
                    print(f"✅ 已清理临时目录：{dir_name}")
                except Exception as e:
                    print(f"⚠️  清理临时目录失败：{e}")


    import atexit

    atexit.register(cleanup_temp_files)