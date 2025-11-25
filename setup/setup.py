# 크롬 브라우저, 파이썬 환경, Selenium, Pandas 등 필수 라이브러리 setup파일입니다.
# Colab 환경에서 크롬드라이버 및 웹 크롤링에 필요한 설정을 적용합니다.


# Colab에 Chrome 설치 / 주석 해제 후 실행
!apt-get update
!apt-get install -y wget unzip
!wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
!dpkg -i google-chrome-stable_current_amd64.deb || apt-get -fy install

# Selenium 및 Python 라이브러리 설치 / 주석 해제 후 실행
!pip install -q selenium
!pip install -q webdriver-manager
!pip install -q beautifulsoup4 pandas numpy scikit-learn matplotlib seaborn requests openpyxl tqdm

# Selenium 및 라이브러리 모듈
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime
import os
import re
import shutil
import tempfile
from tqdm import tqdm

# Chrome 드라이버 설정
user_data_dir = tempfile.mkdtemp()
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
chrome_options.binary_location = "/usr/bin/google-chrome"

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

print("setup complete")