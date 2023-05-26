<div><a href="https://github.com/akkrn/test_openai_api/blob/main/README.md" ><img alt="ru" src="https://img.shields.io/badge/version-on%20english-white"/></a></div>

<details open><summary><h2>📚 Описание</h2></summary>
Данный проект создан для автоматического анализа отзывов на основе их эмоциональной окраски. На вход подается таблица с отзывами, с помощью OpenAI API они анализируются, им выставляется рейтинг и на выходе мы получаем упорядоченную таблицу с выставленным рейтингом для каждого отзыва

</details>

<details><summary><h2>🛠️ Стэк технологий</h2></summary>
<img src="https://img.shields.io/badge/Python-%2314354c.svg?logo=Python&logoColor=white&style=flat" alt="Python" /> <img src="https://img.shields.io/badge/ChatGPT-%23000000.svg?style=flat&logo=openai&logoColor=white" alt="ChatGPT" /> <img src="https://img.shields.io/badge/Pandas-2C2D72?style=flat&logo=pandas&logoColor=white" alt="Pandas" />
</details>
<details><summary><h2>🏗️ Развертывание</h2></summary>
Клонировать репозиторий и перейти в него в командной строке:

```
git clone https://github.com/akkrn/test_openai_api.git
```

Cоздать и активировать виртуальное окружение:

```
python3 -m venv venv
```

* Если у вас Linux/macOS

    ```
    source venv/bin/activate
    ```

* Если у вас windows

    ```
    source venv/Scripts/activate
    ```

```
python3 -m pip install --upgrade pip
```

Установить зависимости из файла requirements.txt:

```
pip install -r requirements.txt
```

Создать .env файл, в который добавить ваш OpenAI AIP ключ 

Запустить проект:

```
python3 main.py
```

</details>
