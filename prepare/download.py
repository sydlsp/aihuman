#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
'''
pip install youtube-dl==2020.12.12
'''
import subprocess
import pandas as pd

def download_youtube_video(video_url, output_path):
    """
    :param video_url: youtube video url
    :param output_dir: file path to save
    """
    # video_url, output_path = info
    try:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # download command
        command = ['yt-dlp',
                   # '--cookies-from-browser', 'chrome', # chrome浏览器'
                   '--cookies', 'D:\研一学习\code\mycode\prepare\www.youtube.com_cookies (2).txt',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        '--merge-output-format','mp4',
        '--output', output_path ,
        video_url]

        # command = ['yt-dlp',
        #            # '--cookies-from-browser', 'chrome', # chrome浏览器'
        #            '--cookies', 'D:\研一学习\code\mycode\prepare\EMTD_dataset\www.youtube.com_cookies.txt',
        #            '-f', 'bestvideo[height=480][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        #            '--merge-output-format', 'mp4',
        #            '--output', output_path,
        #            '--progress',  # 加不加没什么用
        #            video_url]
        # subprocess.run
        result = subprocess.run(command, capture_output=True, text=True, encoding='gbk')  # utf-8  windows下使用gbk

        # pip install -U yt-dlp
        # print("________________________________________________________________")
        # command = ['yt-dlp',
        #            # '--cookies-from-browser', 'chrome', # chrome浏览器
        #            '--cookies', 'D:\研一学习\code\mycode\prepare\EMTD_dataset\www.youtube.com_cookies.txt'
        #            '-f', '"bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"',
        #            '--merge-output-format', 'mp4',
        #            '--output', f'"{output_path}"',
        #            video_url]
        # print("__________________________________________________________________")
        # result = subprocess.run(command,text=True, encoding='utf-8')
        # print("________________________________________________________________________")
        if result.returncode == 0:
            print('Download {:s} successfully!'.format(video_url))
        else:
            print("Fail to download {:s}, error info:\n{:s}".format(video_url, result.stderr))
    except Exception as e:
        print(f"error: {e}")

if __name__ == '__main__':
    df = pd.read_csv("D:\研一学习\code\mycode\prepare\\video_list_all.csv")  # 这里用的时候要改
    save_dir = "G:\\video_src_new"
    # print(os.path.exists(save_dir))
    urls = list(set(df['URL']))
    urls=sorted(urls)
    print(urls)

    print("ok")
    video_output_paths = [os.path.join(save_dir, f"{index}"+".mp4") for index,url in enumerate(urls)]

    for video_url, output_path in zip(urls, video_output_paths):
        download_youtube_video(video_url, output_path)
