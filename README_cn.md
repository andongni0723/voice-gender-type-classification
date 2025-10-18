# Voice Gender Type Classification

> 使用深度學習模型對語音性別進行分類。

## 技術棧 
* 語音預處理 
* 數據標註
* 模型訓練

## 項目結構

```
voice-gender-type-classification/
├── .venv/              # Virtural environment (managed by uv)
├── data/               # Dataset files (ignored by git)
├── tools/
│   └── labeling
├── src/                # Source code for the project
│   ├── train.py
│   └── model.py
├── app.py
├── main.py             # Main script for quick tests
├── pyproject.toml      # Project dependence
└── README.md           # You are here
```

## 運行項目

Fist, clone the Repository to your device.

```bash
git clone https://github.com/<your-name>/voice-gender-type-classification.git
cd voice-gender-type-classification
```

This project uses `uv` as the package manager, \
please make sure your already installed it.

* MacOS / Linux
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

* Window
    ```bash
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

This command will create a virtual environment (`.venv`) and install all required packages specified in `pyproject.toml`.

```bash
uv sync
```

## Usage

TO run the main script and verify the setup, please execute the command:

```bash
uv run main.py
```

and you will see the message in your terminal, which means your setup correctly!

```
Hello from voice-gender-type-classification!
```