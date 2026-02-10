<h1 align="center">MV-Maker</h1>

上传文案+音频+多张图片，即可一键拥有带图文水印的轮播短视频
<h4>WEB UI</h4>
<img width="1920" height="1030" alt="image" src="https://github.com/user-attachments/assets/1c323ece-30a4-4a89-9b93-0dc97dfd044f" />

<br>
- 只需提供一个<b>音频</b>和一段<b>字幕</b>、<b>几张图片</b>，就可以快速生成高清短视频。<br>
- 支持在任意位置添加、字体大小、颜色的文字和图片水印。<br>
- 支持自动检测音频的时间轴。<br>
- 支持自动将文案或字幕按照音频时间轴规划，并按时间轴合成到短视频中。<br>
- 支持自定义字幕的显示位置、字体大小、颜色。<br>
- 支持字幕时间全局偏移<br>
- 支持自定义位置、大小、颜色的文字或图片水印添加。<br>
- 支持生成的短视频在线预览、下载。<br>
- 文案语言无限制。<br>

## 功能扩展
- 代码简洁，备注清晰，易于学习和维护，基于gradio原生，支持<b>AP</b>和<b>web界面</b><br>
- 方便扩展大模型文案自动生成、tts语音合成<br>

### 下阶段计划
- TTS 离线情感语音合成模型支持（如：indextts2）<br>
- OpenAI api v1通用接口支持（支持大多数模型api）<br>
- 增加视频上传合成短视频<br>


## 生成视频演示


https://github.com/user-attachments/assets/d2af6227-16d1-4160-837b-c75d1a209eaa



https://github.com/user-attachments/assets/eb306893-7f61-43f7-a7bc-76159e076618

## 运行
- 显卡非必须，CPU **4核** 或以上，内存 **4GB** 或更高
- windows 10 或ubuntu 20+、MacOS 11.0以上
- Python版本：建议Python3.12+
- 依赖：代码运行会自动安装依赖
- 运行命令：
```shell
python3.12 mv_maker.py
```

### 访问Web界面
运行成功会看到：
✅ 服务启动成功，浏览器访问：http://localhost:7860
* Running on local URL:  http://0.0.0.0:7860

### 访问API文档

打开浏览器，访问 http://localhost:7860/?view=api

## 反馈建议
可以提交 [issue](https://github.com/xyds1025/MV-Maker/issues)
