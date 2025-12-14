from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from openai import OpenAI
import functools
import time
import base64
import json
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

url = "https://service.taipower.com.tw/hvcs/"

driver = webdriver.Chrome()

SYSTEM_PROMPT = '為了有效地完成破解驗證碼的任務，請進行以下步驟，最終以JSON格式回傳結果。\n\n# 步驟\n\n1. **獲取驗證碼圖像**：你需要獲得要破解的驗證碼影像，確保持有適當訪問及使用權限。\n2. **圖像處理**：使用圖像處理技術來消除背景噪音，使驗證碼文本更加清晰。這可能包括灰度化和二值化等技術。\n3. **文字識別**：使用光學字符識別（OCR）技術來提取驗證碼中的文字內容。\n4. **驗證與錯誤處理**：檢查提取的結果是否合理，記錄和處理任何OCR可能出現的錯誤。\n5. **結果輸出**：將最終識別出的驗證碼以結構化的JSON格式返回。\n\n# Output Format\n\n請返回以下格式的JSON文字：\n\n```json\n{\n  "captcha_text": "[提取的驗證碼文本]"\n}\n```\n\n# Notes\n\n- 確保在進行任何破解操作時遵循法律規範和服務條款。\n- 精確度會受到圖像質量及複雜度影響。\n- 如無法破解，請提供適當的錯誤消息。'


def retry_on_exception(retries=3, delay=2, exceptions=(Exception,), on_retry=None):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if on_retry:
                        on_retry(e, attempt)
                    if attempt == retries:
                        raise
                    print(f"Retry {attempt}/{retries} after error: {e}")
                    time.sleep(delay)

        return wrapper

    return decorator_retry


@retry_on_exception(
    retries=3,
    delay=2,
    exceptions=(Exception,),
    on_retry=lambda e, attempt: driver.find_element(By.ID, "refreshChptcha").click(),
)
def recognize_captcha():
    image = driver.find_element(By.ID, "captchaImage")
    image.screenshot("captcha.png")

    base64_image = encode_image_to_base64("captcha.png")
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    results = json.loads(response.choices[0].message.content)
    image_result = results.get("captcha_text")
    driver.find_element(By.ID, "ChptchaCode").send_keys(image_result)
    button = driver.find_element(By.ID, "loginbutton")
    button.click()


def input_account(account, driver):
    acount_input = driver.find_element(By.ID, "UserName")
    acount_input.send_keys(account["account"])

    password_input = driver.find_element(By.ID, "Password")
    password_input.send_keys(account["password"])


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def main():
    driver.get(url)

    with open("account.json", "r") as f:
        account = json.load(f)

    input_account(account, driver)
    recognize_captcha()
    time.sleep(10)


if __name__ == "__main__":
    main()
