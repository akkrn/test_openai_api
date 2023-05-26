<div><a href="https://github.com/akkrn/test_openai_api/blob/main/README-rus.md" ><img alt="ru" src="https://img.shields.io/badge/%D0%B2%D0%B5%D1%80%D1%81%D0%B8%D1%8F-%D0%BD%D0%B0%20%D1%80%D1%83%D1%81%D1%81%D0%BA%D0%BE%D0%BC-white"/></a></div>
<details open><summary><h2>📚 Description</h2></summary>
This project is designed to automatically analyze reviews based on their emotional coloring. The input is a table with reviews, they are analyzed using OpenAI API, assigned a rating and the output is an ordered table with the assigned rating for each review

</details>

<details><summary><h2>🛠️ Tech Stack</h2></summary>
<img src="https://img.shields.io/badge/Python-%2314354c.svg?logo=Python&logoColor=white&style=flat" alt="Python" /> <img src="https://img.shields.io/badge/ChatGPT-%23000000.svg?style=flat&logo=openai&logoColor=white" alt="ChatGPT" /> <img src="https://img.shields.io/badge/Pandas-2C2D72?style=flat&logo=pandas&logoColor=white" alt="Pandas" />
</details>
</details>
<details><summary><h2>🏗️ Installation</h2></summary>

```
git clone https://github.com/akkrn/test_openai_api.git
```

Create and activate a virtual environment:

```
python3 -m venv venv
```

* If you have Linux/macOS

    ```
    source venv/bin/activate
    ```

* If you have windows

    ```
    source venv/Scripts/activate
    ```

```
python3 -m pip install --upgrade pip
```

Install dependencies from the requirements.txt file:

```
pip install -r requirements.txt
```

Create an .env file, in which you add your OpenAI AIP key 

Run the project:

```
python3 main.py
```

</details>
